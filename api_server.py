"""
FlexGrid Optimization API Server
Deploy this on Render.com (or any Python host).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import math

app = FastAPI(title="FlexGrid Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schemas ──────────────────────────────────────────────────────────────

class EVData(BaseModel):
    id: str
    arrival_slot: int
    departure_slot: int
    soc_init_kwh: float
    soc_required_kwh: float
    capacity_kwh: float
    p_max_kw: float
    eta: float = 0.95
    c_deg_eur_per_kwh: float = 0.02


class ScenarioData(BaseModel):
    id: str
    probability: float
    load_deviation_kw: Dict[str, float] = {}
    ev_arrivals: Optional[Dict[str, Any]] = None
    ev_departures: Optional[Dict[str, Any]] = None
    ev_soc_required: Optional[Dict[str, Any]] = None
    ev_soc_init: Optional[Dict[str, Any]] = None


class OptimizeRequest(BaseModel):
    num_time_slots: int
    delta_t_hours: float = 0.25
    t0: int = 0
    site_power_max_kw: float = 300.0
    slack_penalty_eur: float = 10000.0
    smoothing_coeff_eur: float = 1.0
    evs_data: List[EVData]
    scenarios_data: List[ScenarioData]
    dam_profile_kw: Dict[str, float]
    imbalance_price_eur: Dict[str, float]
    intraday_price_eur: Dict[str, float]


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "FlexGrid Optimization API"}


# ── Main optimization endpoint ─────────────────────────────────────────────────

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    """
    Two-stage stochastic LP for EV charging scheduling.
    Uses HiGHS solver via Pyomo.
    """
    try:
        result = run_stochastic_optimization(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_stochastic_optimization(req: OptimizeRequest) -> dict:
    import pyomo.environ as pyo

    T = list(range(req.num_time_slots))
    dt = req.delta_t_hours
    t0 = req.t0
    EVs = [ev.id for ev in req.evs_data]
    ev_map = {ev.id: ev for ev in req.evs_data}
    S = [sc.id for sc in req.scenarios_data]
    sc_map = {sc.id: sc for sc in req.scenarios_data}

    dam = {int(k): v for k, v in req.dam_profile_kw.items()}
    imb_price = {int(k): v for k, v in req.imbalance_price_eur.items()}
    id_price = {int(k): v for k, v in req.intraday_price_eur.items()}

    model = pyo.ConcreteModel()

    # Sets
    model.T = pyo.Set(initialize=T)
    model.EVs = pyo.Set(initialize=EVs)
    model.S = pyo.Set(initialize=S)

    # First-stage: p_now[ev] = power at t0 (fixed across all scenarios)
    model.p_now = pyo.Var(model.EVs, within=pyo.NonNegativeReals)

    # Second-stage: p[ev, t, s], soc[ev, t, s], delta_pos[t, s], delta_neg[t, s]
    model.p = pyo.Var(model.EVs, model.T, model.S, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.EVs, model.T, model.S, within=pyo.NonNegativeReals)
    model.delta_pos = pyo.Var(model.T, model.S, within=pyo.NonNegativeReals)
    model.delta_neg = pyo.Var(model.T, model.S, within=pyo.NonNegativeReals)
    model.soc_slack = pyo.Var(model.EVs, model.S, within=pyo.NonNegativeReals)

    # Objective
    def obj_rule(m):
        cost = 0
        for s in S:
            prob = sc_map[s].probability
            load_dev = {int(k): v for k, v in sc_map[s].load_deviation_kw.items()}
            for t in T:
                lam_imb = imb_price.get(t, 0.09)
                lam_id = id_price.get(t, 0.05)
                cost += prob * (lam_imb * (m.delta_pos[t, s] + m.delta_neg[t, s]) * dt
                                + lam_id * (m.delta_pos[t, s] + m.delta_neg[t, s]) * dt)
            for ev_id in EVs:
                ev = ev_map[ev_id]
                for t in T:
                    cost += prob * ev.c_deg_eur_per_kwh * m.p[ev_id, t, s] * dt
                cost += prob * req.slack_penalty_eur * m.soc_slack[ev_id, s]
        # Smoothing: penalise changes between consecutive slots (first stage anchor)
        for ev_id in EVs:
            for t in T[1:]:
                for s in S:
                    prob = sc_map[s].probability
                    cost += prob * req.smoothing_coeff_eur * (m.p[ev_id, t, s] - m.p[ev_id, t - 1, s]) ** 2
        return cost

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints
    model.constrs = pyo.ConstraintList()

    for s in S:
        load_dev = {int(k): v for k, v in sc_map[s].load_deviation_kw.items()}
        for t in T:
            total_p = sum(model.p[ev_id, t, s] for ev_id in EVs)
            base_load = dam.get(t, 0) + load_dev.get(t, 0)
            model.constrs.add(total_p - base_load == model.delta_pos[t, s] - model.delta_neg[t, s])

        for ev_id in EVs:
            ev = ev_map[ev_id]
            for t in T:
                active = 1 if ev.arrival_slot <= t < ev.departure_slot else 0
                model.constrs.add(model.p[ev_id, t, s] <= active * ev.p_max_kw)
                model.constrs.add(model.p[ev_id, t, s] >= 0)

                # Site power limit
                if t == T[0]:
                    pass  # handled by site-level below

            # SOC dynamics
            for t in T:
                if t == 0:
                    soc_prev = ev.soc_init_kwh
                else:
                    soc_prev = model.soc[ev_id, t - 1, s]
                model.constrs.add(
                    model.soc[ev_id, t, s] == soc_prev + ev.eta * model.p[ev_id, t, s] * dt
                )
                model.constrs.add(model.soc[ev_id, t, s] <= ev.capacity_kwh)

            # Final SOC requirement (with slack)
            dep = ev.departure_slot - 1
            dep = min(dep, req.num_time_slots - 1)
            model.constrs.add(
                model.soc[ev_id, dep, s] + model.soc_slack[ev_id, s] >= ev.soc_required_kwh
            )

        # Site power limit per slot
        for t in T:
            model.constrs.add(
                sum(model.p[ev_id, t, s] for ev_id in EVs) <= req.site_power_max_kw
            )

    # First-stage non-anticipativity: p[ev, t0, s] = p_now[ev] for all s
    for ev_id in EVs:
        for s in S:
            model.constrs.add(model.p[ev_id, t0, s] == model.p_now[ev_id])

    # Solve using appsi_highs (requires highspy package)
    solver = pyo.SolverFactory('appsi_highs')
    results = solver.solve(model)

    if results.termination_condition not in [
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    ]:
        raise Exception(f"Solver status: {results.termination_condition}")

    # Extract results
    obj_val = pyo.value(model.obj)

    # p_now
    p_now = {ev_id: round(pyo.value(model.p_now[ev_id]), 4) for ev_id in EVs}

    # Schedule summary (average across scenarios)
    schedule_summary = []
    for t in T:
        total_p = sum(
            sc_map[s].probability * pyo.value(model.p[ev_id, t, s])
            for s in S for ev_id in EVs
        )
        dam_t = dam.get(t, 0)
        dp = sum(sc_map[s].probability * pyo.value(model.delta_pos[t, s]) for s in S)
        dn = sum(sc_map[s].probability * pyo.value(model.delta_neg[t, s]) for s in S)
        schedule_summary.append({
            "slot": t,
            "total_power_kw": round(total_p, 4),
            "dam_kw": dam_t,
            "delta_pos_kw": round(dp, 4),
            "delta_neg_kw": round(dn, 4),
        })

    # SOC trajectories + power per slot (avg across scenarios)
    soc_trajectories = []
    for ev_id in EVs:
        traj = {"ev_id": ev_id, "soc": [], "p_kw": []}
        for t in T:
            avg_soc = sum(sc_map[s].probability * pyo.value(model.soc[ev_id, t, s]) for s in S)
            avg_p = sum(sc_map[s].probability * pyo.value(model.p[ev_id, t, s]) for s in S)
            traj["soc"].append(round(avg_soc, 4))
            traj["p_kw"].append(round(avg_p, 4))
        soc_trajectories.append(traj)

    # Cost breakdown
    dt_ = dt
    cost_imb = sum(
        sc_map[s].probability * imb_price.get(t, 0.09) * (
            pyo.value(model.delta_pos[t, s]) + pyo.value(model.delta_neg[t, s])
        ) * dt_
        for s in S for t in T
    )
    cost_id = sum(
        sc_map[s].probability * id_price.get(t, 0.05) * (
            pyo.value(model.delta_pos[t, s]) + pyo.value(model.delta_neg[t, s])
        ) * dt_
        for s in S for t in T
    )
    cost_deg = sum(
        sc_map[s].probability * ev_map[ev_id].c_deg_eur_per_kwh * pyo.value(model.p[ev_id, t, s]) * dt_
        for s in S for ev_id in EVs for t in T
    )
    cost_slack = sum(
        sc_map[s].probability * req.slack_penalty_eur * pyo.value(model.soc_slack[ev_id, s])
        for s in S for ev_id in EVs
    )
    cost_smooth = max(0, obj_val - cost_imb - cost_id - cost_deg - cost_slack)

    # Baseline (greedy) cost
    baseline = compute_baseline(req, dam, imb_price, id_price)

    savings = baseline["total"] - obj_val
    savings_pct = (savings / baseline["total"] * 100) if baseline["total"] > 0 else 0.0

    return {
        "status": "optimal",
        "result_objective_value": round(obj_val, 4),
        "result_cost_imbalance": round(cost_imb, 4),
        "result_cost_intraday": round(cost_id, 4),
        "result_cost_degradation": round(cost_deg, 4),
        "result_cost_smoothing": round(cost_smooth, 4),
        "result_cost_slack": round(cost_slack, 4),
        "result_p_now_kw": p_now,
        "result_schedule_summary": schedule_summary,
        "result_soc_trajectories": soc_trajectories,
        "baseline_cost_total": round(baseline["total"], 4),
        "baseline_cost_imbalance": round(baseline["imbalance"], 4),
        "baseline_cost_intraday": round(baseline["intraday"], 4),
        "baseline_cost_degradation": round(baseline["degradation"], 4),
        "savings_eur": round(savings, 4),
        "savings_pct": round(savings_pct, 2),
    }


def compute_baseline(req: OptimizeRequest, dam, imb_price, id_price) -> dict:
    """Greedy baseline: each EV charges at max power as soon as it arrives."""
    dt = req.delta_t_hours
    T = list(range(req.num_time_slots))
    ev_states = {ev.id: ev.soc_init_kwh for ev in req.evs_data}
    ev_map = {ev.id: ev for ev in req.evs_data}

    total, cost_imb, cost_id, cost_deg = 0.0, 0.0, 0.0, 0.0

    for t in T:
        total_p = 0.0
        for ev_id, ev in ev_map.items():
            if ev.arrival_slot <= t < ev.departure_slot:
                soc = ev_states[ev_id]
                needed = ev.soc_required_kwh - soc
                if needed > 0.001:
                    p = min(ev.p_max_kw, needed / (ev.eta * dt))
                    ev_states[ev_id] += ev.eta * p * dt
                    total_p += p
                    cost_deg += ev.c_deg_eur_per_kwh * p * dt

        dev = total_p - dam.get(t, 0)
        dp = max(dev, 0)
        dn = max(-dev, 0)
        cost_imb += imb_price.get(t, 0.09) * (dp + dn) * dt
        cost_id += id_price.get(t, 0.05) * (dp + dn) * dt

    total = cost_imb + cost_id + cost_deg
    return {"total": total, "imbalance": cost_imb, "intraday": cost_id, "degradation": cost_deg}
