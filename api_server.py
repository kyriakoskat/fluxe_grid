"""
FlexGrid Optimization API Server
Deploy this on Render.com (or any Python host).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import math
# cvxpy and numpy imported inside run_stochastic_optimization

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
    import cvxpy as cp
    import numpy as np

    T = list(range(req.num_time_slots))
    nT = req.num_time_slots
    dt = req.delta_t_hours
    t0 = req.t0
    EVs = [ev.id for ev in req.evs_data]
    nEV = len(EVs)
    ev_map = {ev.id: ev for ev in req.evs_data}
    S = [sc.id for sc in req.scenarios_data]
    nS = len(S)
    sc_map = {sc.id: sc for sc in req.scenarios_data}

    dam = {int(k): v for k, v in req.dam_profile_kw.items()}
    imb_price = {int(k): v for k, v in req.imbalance_price_eur.items()}
    id_price = {int(k): v for k, v in req.intraday_price_eur.items()}

    # Variables
    # p[ev, t, s], soc[ev, t, s], delta_pos[t,s], delta_neg[t,s], soc_slack[ev,s], p_now[ev]
    p = cp.Variable((nEV, nT, nS), nonneg=True)          # power per EV per slot per scenario
    soc = cp.Variable((nEV, nT, nS), nonneg=True)        # SOC per EV per slot per scenario
    delta_pos = cp.Variable((nT, nS), nonneg=True)
    delta_neg = cp.Variable((nT, nS), nonneg=True)
    soc_slack = cp.Variable((nEV, nS), nonneg=True)
    p_now = cp.Variable(nEV, nonneg=True)                # first-stage decisions

    ev_idx = {ev_id: i for i, ev_id in enumerate(EVs)}
    sc_idx = {s_id: j for j, s_id in enumerate(S)}

    constraints = []
    cost = 0

    for j, s_id in enumerate(S):
        sc = sc_map[s_id]
        prob = sc.probability
        load_dev = {int(k): v for k, v in sc.load_deviation_kw.items()}

        for t in T:
            lam = imb_price.get(t, 0.09) + id_price.get(t, 0.05)
            cost += prob * lam * (delta_pos[t, j] + delta_neg[t, j]) * dt

            # Imbalance definition
            total_p_t = cp.sum(p[:, t, j])
            base_load = dam.get(t, 0) + load_dev.get(t, 0)
            constraints.append(total_p_t - base_load == delta_pos[t, j] - delta_neg[t, j])

            # Site power limit
            constraints.append(total_p_t <= req.site_power_max_kw)

        for i, ev_id in enumerate(EVs):
            ev = ev_map[ev_id]

            cost += prob * req.slack_penalty_eur * soc_slack[i, j]

            for t in T:
                # Degradation cost
                cost += prob * ev.c_deg_eur_per_kwh * p[i, t, j] * dt

                # Power limits
                active = 1 if ev.arrival_slot <= t < ev.departure_slot else 0
                constraints.append(p[i, t, j] <= active * ev.p_max_kw)

                # SOC dynamics
                soc_prev = ev.soc_init_kwh if t == 0 else soc[i, t - 1, j]
                constraints.append(soc[i, t, j] == soc_prev + ev.eta * p[i, t, j] * dt)
                constraints.append(soc[i, t, j] <= ev.capacity_kwh)

            # Final SOC requirement
            dep = min(ev.departure_slot - 1, nT - 1)
            constraints.append(soc[i, dep, j] + soc_slack[i, j] >= ev.soc_required_kwh)

        # Non-anticipativity: p[ev, t0, s] = p_now[ev]
        for i in range(nEV):
            constraints.append(p[i, t0, j] == p_now[i])

    # Quadratic smoothing: sum over scenarios of prob * coeff * (p[t] - p[t-1])^2
    for j, s_id in enumerate(S):
        prob = sc_map[s_id].probability
        for i in range(nEV):
            for t in T[1:]:
                cost += prob * req.smoothing_coeff_eur * cp.square(p[i, t, j] - p[i, t - 1, j])

    prob_obj = cp.Minimize(cost)
    problem = cp.Problem(prob_obj, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception(f"Solver status: {problem.status}")

    obj_val = problem.value

    # p_now
    p_now_val = {ev_id: round(float(p_now.value[i]), 4) for i, ev_id in enumerate(EVs)}

    # Schedule summary
    schedule_summary = []
    for t in T:
        total_p_t = sum(
            sc_map[s_id].probability * float(p.value[i, t, j])
            for j, s_id in enumerate(S)
            for i in range(nEV)
        )
        dp = sum(sc_map[s_id].probability * float(delta_pos.value[t, j]) for j, s_id in enumerate(S))
        dn = sum(sc_map[s_id].probability * float(delta_neg.value[t, j]) for j, s_id in enumerate(S))
        schedule_summary.append({
            "slot": t,
            "total_power_kw": round(total_p_t, 4),
            "dam_kw": dam.get(t, 0),
            "delta_pos_kw": round(dp, 4),
            "delta_neg_kw": round(dn, 4),
        })

    # SOC trajectories
    soc_trajectories = []
    for i, ev_id in enumerate(EVs):
        traj = {"ev_id": ev_id, "soc": [], "p_kw": []}
        for t in T:
            avg_soc = sum(sc_map[s_id].probability * float(soc.value[i, t, j]) for j, s_id in enumerate(S))
            avg_p = sum(sc_map[s_id].probability * float(p.value[i, t, j]) for j, s_id in enumerate(S))
            traj["soc"].append(round(avg_soc, 4))
            traj["p_kw"].append(round(avg_p, 4))
        soc_trajectories.append(traj)

    # Cost breakdown
    cost_imb = sum(
        sc_map[s_id].probability * imb_price.get(t, 0.09) * (
            float(delta_pos.value[t, j]) + float(delta_neg.value[t, j])
        ) * dt
        for j, s_id in enumerate(S) for t in T
    )
    cost_id = sum(
        sc_map[s_id].probability * id_price.get(t, 0.05) * (
            float(delta_pos.value[t, j]) + float(delta_neg.value[t, j])
        ) * dt
        for j, s_id in enumerate(S) for t in T
    )
    cost_deg = sum(
        sc_map[s_id].probability * ev_map[ev_id].c_deg_eur_per_kwh * float(p.value[i, t, j]) * dt
        for j, s_id in enumerate(S) for i, ev_id in enumerate(EVs) for t in T
    )
    cost_slack_val = sum(
        sc_map[s_id].probability * req.slack_penalty_eur * float(soc_slack.value[i, j])
        for j, s_id in enumerate(S) for i in range(nEV)
    )
    cost_smooth = max(0, obj_val - cost_imb - cost_id - cost_deg - cost_slack_val)

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
        "result_cost_slack": round(cost_slack_val, 4),
        "result_p_now_kw": p_now_val,
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
