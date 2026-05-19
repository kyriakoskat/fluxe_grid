"""
FlexGrid Optimization API Server
Deploy this on Render.com (or any Python host).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

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
    EVs = req.evs_data
    nEV = len(EVs)
    ev_map = {ev.id: ev for ev in EVs}
    Sc = req.scenarios_data
    nS = len(Sc)
    sc_map = {sc.id: sc for sc in Sc}

    dam = {int(k): v for k, v in req.dam_profile_kw.items()}
    imb_price = {int(k): v for k, v in req.imbalance_price_eur.items()}
    id_price = {int(k): v for k, v in req.intraday_price_eur.items()}

    # ── Variables (flat 1D per scenario to avoid 3D array issues) ─────────────
    # p_vars[j][i][t] = scalar Variable for EV i, slot t, scenario j
    p_vars = [
        [[cp.Variable(nonneg=True) for _ in T] for _ in range(nEV)]
        for _ in range(nS)
    ]
    soc_vars = [
        [[cp.Variable(nonneg=True) for _ in T] for _ in range(nEV)]
        for _ in range(nS)
    ]
    delta_pos = [[cp.Variable(nonneg=True) for _ in T] for _ in range(nS)]
    delta_neg = [[cp.Variable(nonneg=True) for _ in T] for _ in range(nS)]
    soc_slack = [[cp.Variable(nonneg=True) for _ in range(nEV)] for _ in range(nS)]
    # First-stage: p_now[i]
    p_now = [cp.Variable(nonneg=True) for _ in range(nEV)]

    constraints = []
    cost_terms = []

    for j, sc in enumerate(Sc):
        prob = sc.probability
        load_dev = {int(k): v for k, v in sc.load_deviation_kw.items()}

        for t in T:
            lam = imb_price.get(t, 0.09) + id_price.get(t, 0.05)
            cost_terms.append(prob * lam * (delta_pos[j][t] + delta_neg[j][t]) * dt)

            total_p_t = sum(p_vars[j][i][t] for i in range(nEV))
            base_load = dam.get(t, 0) + load_dev.get(t, 0)
            constraints.append(total_p_t - base_load == delta_pos[j][t] - delta_neg[j][t])
            constraints.append(total_p_t <= req.site_power_max_kw)

        for i, ev in enumerate(EVs):
            cost_terms.append(prob * req.slack_penalty_eur * soc_slack[j][i])

            for t in T:
                cost_terms.append(prob * ev.c_deg_eur_per_kwh * p_vars[j][i][t] * dt)

                active = 1 if ev.arrival_slot <= t < ev.departure_slot else 0
                constraints.append(p_vars[j][i][t] <= active * ev.p_max_kw)

                soc_prev = ev.soc_init_kwh if t == 0 else soc_vars[j][i][t - 1]
                constraints.append(soc_vars[j][i][t] == soc_prev + ev.eta * p_vars[j][i][t] * dt)
                constraints.append(soc_vars[j][i][t] <= ev.capacity_kwh)

                # Quadratic smoothing per EV per slot
                if t > 0:
                    diff = p_vars[j][i][t] - p_vars[j][i][t - 1]
                    cost_terms.append(prob * req.smoothing_coeff_eur * cp.square(diff))

            dep = min(ev.departure_slot - 1, nT - 1)
            constraints.append(soc_vars[j][i][dep] + soc_slack[j][i] >= ev.soc_required_kwh)

        # Non-anticipativity: p[i, t0, j] = p_now[i]
        for i in range(nEV):
            constraints.append(p_vars[j][i][t0] == p_now[i])

    objective = cp.Minimize(sum(cost_terms))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception(f"Solver status: {problem.status}")

    obj_val = problem.value

    # ── Extract results ────────────────────────────────────────────────────────
    def pv(j, i, t): return float(p_vars[j][i][t].value or 0)
    def sv(j, i, t): return float(soc_vars[j][i][t].value or 0)
    def dpv(j, t): return float(delta_pos[j][t].value or 0)
    def dnv(j, t): return float(delta_neg[j][t].value or 0)
    def slv(j, i): return float(soc_slack[j][i].value or 0)

    p_now_val = {ev.id: round(float(p_now[i].value or 0), 4) for i, ev in enumerate(EVs)}

    schedule_summary = []
    for t in T:
        total_p_t = sum(Sc[j].probability * pv(j, i, t) for j in range(nS) for i in range(nEV))
        dp = sum(Sc[j].probability * dpv(j, t) for j in range(nS))
        dn = sum(Sc[j].probability * dnv(j, t) for j in range(nS))
        schedule_summary.append({
            "slot": t,
            "total_power_kw": round(total_p_t, 4),
            "dam_kw": dam.get(t, 0),
            "delta_pos_kw": round(dp, 4),
            "delta_neg_kw": round(dn, 4),
        })

    soc_trajectories = []
    for i, ev in enumerate(EVs):
        traj = {"ev_id": ev.id, "soc": [], "p_kw": []}
        for t in T:
            avg_soc = sum(Sc[j].probability * sv(j, i, t) for j in range(nS))
            avg_p = sum(Sc[j].probability * pv(j, i, t) for j in range(nS))
            traj["soc"].append(round(avg_soc, 4))
            traj["p_kw"].append(round(avg_p, 4))
        soc_trajectories.append(traj)

    cost_imb = sum(Sc[j].probability * imb_price.get(t, 0.09) * (dpv(j, t) + dnv(j, t)) * dt for j in range(nS) for t in T)
    cost_id  = sum(Sc[j].probability * id_price.get(t, 0.05) * (dpv(j, t) + dnv(j, t)) * dt for j in range(nS) for t in T)
    cost_deg = sum(Sc[j].probability * EVs[i].c_deg_eur_per_kwh * pv(j, i, t) * dt for j in range(nS) for i in range(nEV) for t in T)
    cost_slack_val = sum(Sc[j].probability * req.slack_penalty_eur * slv(j, i) for j in range(nS) for i in range(nEV))
    cost_smooth = sum(
        Sc[j].probability * req.smoothing_coeff_eur * (pv(j, i, t) - pv(j, i, t - 1)) ** 2
        for j in range(nS) for i in range(nEV) for t in T[1:]
    )

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
    dt = req.delta_t_hours
    T = list(range(req.num_time_slots))
    ev_states = {ev.id: ev.soc_init_kwh for ev in req.evs_data}

    total, cost_imb, cost_id, cost_deg = 0.0, 0.0, 0.0, 0.0

    for t in T:
        total_p = 0.0
        for ev in req.evs_data:
            if ev.arrival_slot <= t < ev.departure_slot:
                needed = ev.soc_required_kwh - ev_states[ev.id]
                if needed > 0.001:
                    p = min(ev.p_max_kw, needed / (ev.eta * dt))
                    ev_states[ev.id] += ev.eta * p * dt
                    total_p += p
                    cost_deg += ev.c_deg_eur_per_kwh * p * dt

        dev = total_p - dam.get(t, 0)
        dp = max(dev, 0)
        dn = max(-dev, 0)
        cost_imb += imb_price.get(t, 0.09) * (dp + dn) * dt
        cost_id += id_price.get(t, 0.05) * (dp + dn) * dt

    total = cost_imb + cost_id + cost_deg
    return {"total": total, "imbalance": cost_imb, "intraday": cost_id, "degradation": cost_deg}
