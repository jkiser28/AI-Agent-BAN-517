"""
Supply Chain Analytics Inventory Tools
======================================
Deterministic, pure-Python implementations of fundamental inventory decision models:
1. Economic Order Quantity (EOQ): calculate_eoq()
2. Quantity Discount Model: calculate_quantity_discount()
3. Safety Stock & Reorder Point (Continuous Review): calculate_rop()
4. Fixed-Period Review Model (Periodic Review): calculate_periodic_review()
5. Newsvendor Model (Single-Period): calculate_newsvendor()

Every tool strictly satisfies:
- Input validation with strict type/bound checking
- Returning structured results containing:
  * inputs (validated parameter dict, never inventing missing inputs)
  * intermediate_calculations (explicit step-by-step math metrics)
  * final_answer (primary decision recommendation)
  * assumptions (explicit list of model assumptions)
- Returning top-level backward-compatible keys
- Pure Python numerical calculations using standard library math & statistics
"""

import math
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union


# ===========================================================================
# 0. Result Structure & Statistical Helpers
# ===========================================================================

class InventoryResult(dict):
    """
    Structured result dictionary that supports:
    1. Structured sub-dictionaries:
       - result["inputs"]
       - result["intermediate_calculations"]
       - result["final_answer"]
       - result["assumptions"]
    2. Direct attribute access:
       - result.inputs
       - result.intermediate_calculations
       - result.final_answer
       - result.assumptions
    3. Top-level keys for seamless backward compatibility with existing tests and agents:
       - result["eoq"], result["reorder_point"], result["safety_stock"], etc.
    """
    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(f"'InventoryResult' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def standard_normal_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution N(0, 1)."""
    return statistics.NormalDist(0.0, 1.0).cdf(z)


def standard_normal_ppf(p: float) -> float:
    """Percent point function (inverse CDF / quantile) for standard normal N(0, 1)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability must be strictly between 0 and 1, got {p}")
    return statistics.NormalDist(0.0, 1.0).inv_cdf(p)


def standard_normal_pdf(z: float) -> float:
    """Probability density function for standard normal distribution N(0, 1)."""
    return statistics.NormalDist(0.0, 1.0).pdf(z)


def unit_normal_loss(z: float) -> float:
    """
    Unit normal loss function:
    L(z) = E[max(0, Z - z)] = phi(z) - z * (1 - Phi(z))
    Used for calculating expected lost sales in inventory theory.
    """
    phi_z = standard_normal_pdf(z)
    Phi_z = standard_normal_cdf(z)
    return phi_z - z * (1.0 - Phi_z)


# ===========================================================================
# 1. Economic Order Quantity (EOQ) Model
# ===========================================================================

def calculate_eoq(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_price: Optional[float] = None,
    holding_rate: Optional[float] = None,
    working_days_per_year: int = 250,
    lead_time_days: Optional[float] = None,
) -> InventoryResult:
    """
    Calculate Economic Order Quantity (EOQ): Q = sqrt(2 * D * S / H).

    Validates inputs, executes intermediate cost trade-off calculations,
    and returns a structured result with explicit assumptions.
    Never invents missing numerical inputs.
    """
    # 1. Input Validation (never invent missing numerical inputs)
    if annual_demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' (D) must be provided.")
    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' (S) must be provided.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"Annual demand must be a positive number > 0, got {annual_demand}")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"Order cost must be a positive number > 0, got {order_cost}")

    D = float(annual_demand)
    S = float(order_cost)

    # Validate and resolve holding cost H
    if holding_cost is not None:
        if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
            raise ValueError(f"Holding cost must be > 0, got {holding_cost}")
        H = float(holding_cost)
        holding_source = "explicit_holding_cost"
    elif unit_price is not None and holding_rate is not None:
        if not isinstance(unit_price, (int, float)) or unit_price <= 0:
            raise ValueError(f"Unit price must be > 0, got {unit_price}")
        if not isinstance(holding_rate, (int, float)) or holding_rate <= 0:
            raise ValueError(f"Holding rate must be > 0, got {holding_rate}")
        rate = float(holding_rate)
        if rate > 1.0:
            rate = rate / 100.0  # handle percentage passed as 20 instead of 0.20
        H = float(unit_price) * rate
        holding_source = "unit_price_times_holding_rate"
    else:
        raise ValueError(
            "Missing required numerical input: must provide either 'holding_cost' ($/unit/year) "
            "or both 'unit_price' and 'holding_rate'."
        )

    if working_days_per_year <= 0:
        raise ValueError(f"Working days per year must be > 0, got {working_days_per_year}")

    # 2. Mathematical Calculations
    # Formula: Q = sqrt(2 * D * S / H)
    eoq = math.sqrt((2.0 * D * S) / H)
    orders_per_year = D / eoq
    cycle_time_years = eoq / D
    cycle_time_days = cycle_time_years * working_days_per_year

    annual_ordering_cost = orders_per_year * S
    annual_holding_cost = (eoq / 2.0) * H
    total_inventory_cost = annual_ordering_cost + annual_holding_cost
    cost_parity_difference = abs(annual_ordering_cost - annual_holding_cost)

    total_annual_cost = None
    annual_purchase_cost = None
    if unit_price is not None:
        annual_purchase_cost = D * float(unit_price)
        total_annual_cost = annual_purchase_cost + total_inventory_cost

    reorder_point = None
    daily_demand = None
    if lead_time_days is not None:
        if lead_time_days < 0:
            raise ValueError(f"Lead time days cannot be negative, got {lead_time_days}")
        daily_demand = D / working_days_per_year
        reorder_point = daily_demand * float(lead_time_days)

    # 3. Model Assumptions
    assumptions = [
        "Annual demand rate D is constant, continuous, and known with certainty.",
        "Fixed ordering cost S is independent of order batch size.",
        "Inventory carrying/holding cost H is linear and constant per unit per year.",
        "Replenishment lead time is deterministic; order quantity Q arrives in a single batch.",
        "No shortages, stockouts, or backorders are permitted.",
        "No quantity discounts apply in standard EOQ; unit price is constant."
    ]

    # 4. Structured Output
    inputs_dict = {
        "annual_demand": D,
        "order_cost": S,
        "holding_cost": holding_cost,
        "unit_price": unit_price,
        "holding_rate": holding_rate,
        "working_days_per_year": working_days_per_year,
        "lead_time_days": lead_time_days,
        "holding_cost_derivation": holding_source,
    }

    intermediates_dict = {
        "unit_holding_cost_H": round(H, 4),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_years": round(cycle_time_years, 4),
        "cycle_time_days": round(cycle_time_days, 1),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 2) if total_annual_cost is not None else None,
        "cost_parity_difference": round(cost_parity_difference, 4),
        "daily_demand": round(daily_demand, 2) if daily_demand is not None else None,
        "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
    }

    final_answer_dict = {
        "optimal_order_quantity_Q": round(eoq, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "orders_per_year": round(orders_per_year, 2),
        "order_cycle_days": round(cycle_time_days, 1),
        "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
    }

    res = InventoryResult({
        "tool": "calculate_eoq",
        "inputs": inputs_dict,
        "intermediate_calculations": intermediates_dict,
        "final_answer": final_answer_dict,
        "assumptions": assumptions,
        # Backward-compatible convenience keys:
        "eoq": round(eoq, 2),
        "optimal_order_quantity": round(eoq, 2),
        "annual_demand": round(D, 4),
        "order_cost": round(S, 4),
        "holding_cost_per_unit": round(H, 4),
        "unit_price": round(unit_price, 4) if unit_price is not None else None,
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_years": round(cycle_time_years, 4),
        "cycle_time_days": round(cycle_time_days, 1),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2) if total_annual_cost is not None else None,
        "reorder_point": round(reorder_point, 2) if reorder_point is not None else None,
        "lead_time_days": lead_time_days,
        "working_days": working_days_per_year,
    })
    return res


# ===========================================================================
# 2. Quantity Discount Model
# ===========================================================================

@dataclass
class DiscountTierEvaluation:
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    unconstrained_eoq: float
    is_eoq_feasible: bool
    candidate_qty: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    is_optimal: bool = False


def calculate_quantity_discount(
    annual_demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    discount_schedule: Optional[List[Dict[str, Any]]] = None,
    holding_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
) -> InventoryResult:
    """
    Evaluate all price break tiers to find the global cost-minimizing order quantity.
    Supports both fixed holding cost and percentage holding cost.
    Validates inputs, evaluates feasible quantities, compares total annual cost,
    and returns a structured result with explicit assumptions.
    Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if annual_demand is None:
        raise ValueError("Missing required numerical input: 'annual_demand' must be provided.")
    if order_cost is None:
        raise ValueError("Missing required numerical input: 'order_cost' must be provided.")
    if not discount_schedule:
        raise ValueError("Missing required numerical input: 'discount_schedule' cannot be empty.")
    if holding_cost is None and holding_rate is None:
        raise ValueError("Missing required numerical input: must provide either 'holding_cost' or 'holding_rate'.")

    if not isinstance(annual_demand, (int, float)) or annual_demand <= 0:
        raise ValueError(f"Annual demand must be > 0, got {annual_demand}")
    if not isinstance(order_cost, (int, float)) or order_cost <= 0:
        raise ValueError(f"Order cost must be > 0, got {order_cost}")

    D = float(annual_demand)
    S = float(order_cost)

    rate = None
    if holding_rate is not None:
        if not isinstance(holding_rate, (int, float)) or holding_rate <= 0:
            raise ValueError(f"Holding rate must be > 0, got {holding_rate}")
        rate = float(holding_rate)
        if rate > 1.0:
            rate = rate / 100.0

    fixed_h = None
    if holding_cost is not None:
        if not isinstance(holding_cost, (int, float)) or holding_cost <= 0:
            raise ValueError(f"Fixed holding cost must be > 0, got {holding_cost}")
        fixed_h = float(holding_cost)

    # Sort schedule by min_qty
    sorted_tiers = sorted(discount_schedule, key=lambda x: x.get("min_qty", 0))

    evaluations: List[DiscountTierEvaluation] = []
    best_candidate_idx = -1
    lowest_total_cost = float("inf")

    # 2. Tier Evaluations
    for idx, tier in enumerate(sorted_tiers):
        if "min_qty" not in tier or "price" not in tier:
            raise ValueError(f"Tier {idx} must define 'min_qty' and 'price'")

        min_q = float(tier["min_qty"])
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None
        price = float(tier["price"])

        if min_q < 0:
            raise ValueError(f"min_qty cannot be negative in tier {idx}")
        if price <= 0:
            raise ValueError(f"Price must be > 0, got {price} in tier {idx}")

        # Compute holding cost for this tier
        if rate is not None:
            h_tier = price * rate
        else:
            h_tier = fixed_h  # type: ignore

        if h_tier <= 0:
            raise ValueError(f"Effective holding cost must be > 0 in tier {idx}")

        # Unconstrained EOQ for this tier
        unconstrained_eoq = math.sqrt((2.0 * D * S) / h_tier)

        # Feasibility check for all-units discount:
        # Since TC curve is convex with minimum at unconstrained_eoq:
        # 1. If EOQ in [min_q, max_q], optimal feasible quantity is EOQ
        # 2. If EOQ < min_q, optimal feasible quantity is price break floor min_q
        # 3. If EOQ > max_q, bounded at max_q (or infeasible for lower price tier)
        if min_q <= unconstrained_eoq and (max_q is None or unconstrained_eoq <= max_q):
            candidate_q = unconstrained_eoq
            is_feasible = True
        elif unconstrained_eoq < min_q:
            candidate_q = min_q
            is_feasible = False  # adjusted to price break threshold
        else:
            candidate_q = max_q if max_q is not None else unconstrained_eoq
            is_feasible = False  # constrained at ceiling

        purchase_cost = D * price
        ordering_cost = (D / candidate_q) * S
        holding_c = (candidate_q / 2.0) * h_tier
        total_c = purchase_cost + ordering_cost + holding_c

        eval_item = DiscountTierEvaluation(
            tier_index=idx + 1,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=price,
            holding_cost_per_unit=round(h_tier, 4),
            unconstrained_eoq=round(unconstrained_eoq, 2),
            is_eoq_feasible=round(unconstrained_eoq, 2) == round(candidate_q, 2),
            candidate_qty=round(candidate_q, 2),
            annual_purchase_cost=round(purchase_cost, 2),
            annual_ordering_cost=round(ordering_cost, 2),
            annual_holding_cost=round(holding_c, 2),
            total_annual_cost=round(total_c, 2),
        )
        evaluations.append(eval_item)

        if total_c < lowest_total_cost:
            lowest_total_cost = total_c
            best_candidate_idx = len(evaluations) - 1

    if best_candidate_idx == -1:
        raise ValueError("Could not find any feasible order quantity across discount tiers")

    evaluations[best_candidate_idx].is_optimal = True
    best_eval = evaluations[best_candidate_idx]

    # Baseline cost without discount (Tier 1 unconstrained EOQ)
    base_tier = sorted_tiers[0]
    base_p = float(base_tier["price"])
    base_h = base_p * rate if rate is not None else fixed_h  # type: ignore
    base_eoq = math.sqrt((2.0 * D * S) / base_h)
    baseline_total_cost = (D * base_p) + (D / base_eoq) * S + (base_eoq / 2.0) * base_h
    annual_savings = max(0.0, baseline_total_cost - best_eval.total_annual_cost)
    pct_savings = (annual_savings / baseline_total_cost * 100.0) if baseline_total_cost > 0 else 0.0

    # 3. Model Assumptions
    assumptions = [
        "All-units quantity discount schedule: the discounted price applies to every unit in the order.",
        "Holding cost can be constant across tiers or proportional to unit purchase price (H_i = h * C_i).",
        "Demand rate D and setup cost S remain constant across all order quantities.",
        "Candidate order quantities are evaluated at unconstrained EOQs (if feasible) or price break boundaries.",
        "Warehouse capacity and operating capital are adequate to support larger batch sizes."
    ]

    # 4. Structured Output
    inputs_dict = {
        "annual_demand": D,
        "order_cost": S,
        "holding_cost": fixed_h,
        "holding_rate": rate,
        "discount_schedule": discount_schedule,
        "holding_cost_mode": "percentage_of_price" if rate is not None else "fixed_per_unit"
    }

    intermediates_dict = {
        "evaluated_tiers": [asdict(e) for e in evaluations],
        "baseline_tier_price": base_p,
        "baseline_eoq": round(base_eoq, 2),
        "baseline_total_annual_cost": round(baseline_total_cost, 2),
        "annual_savings_vs_base": round(annual_savings, 2),
        "percentage_savings": round(pct_savings, 2),
    }

    final_answer_dict = {
        "optimal_order_quantity": best_eval.candidate_qty,
        "optimal_tier": best_eval.tier_index,
        "optimal_unit_price": best_eval.unit_price,
        "optimal_total_cost": best_eval.total_annual_cost,
        "annual_savings": round(annual_savings, 2),
    }

    res = InventoryResult({
        "tool": "calculate_quantity_discount",
        "inputs": inputs_dict,
        "intermediate_calculations": intermediates_dict,
        "final_answer": final_answer_dict,
        "assumptions": assumptions,
        # Backward-compatible convenience keys:
        "annual_demand": D,
        "order_cost": S,
        "holding_rate": rate,
        "holding_cost": fixed_h,
        "optimal_order_quantity": best_eval.candidate_qty,
        "optimal_tier": best_eval.tier_index,
        "optimal_unit_price": best_eval.unit_price,
        "optimal_total_cost": best_eval.total_annual_cost,
        "annual_savings_vs_base": round(annual_savings, 2),
        "percentage_savings": round(pct_savings, 2),
        "evaluated_tiers": [asdict(e) for e in evaluations],
    })
    return res


# ===========================================================================
# 3. Safety Stock & Reorder Point (ROP) Model
# ===========================================================================

def calculate_rop(
    service_level: Optional[float] = 0.95,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    lead_time: Optional[float] = None,
    lead_time_std: float = 0.0,
    holding_cost_per_unit: Optional[float] = None,
    # Aliases for backward compatibility:
    daily_demand_mean: Optional[float] = None,
    daily_demand_std: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    lead_time_std_days: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
) -> InventoryResult:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP).

    Supports all 4 demand and lead time variability combinations:
    1. constant demand and constant lead time (sigma_d = 0, sigma_L = 0)
    2. variable demand and constant lead time (sigma_d > 0, sigma_L = 0)
    3. constant demand and variable lead time (sigma_d = 0, sigma_L > 0)
    4. variable demand and variable lead time (sigma_d > 0, sigma_L > 0):
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Validates inputs, executes intermediate calculations, and returns structured result with assumptions.
    Never invents missing numerical inputs.
    """
    # 1. Resolve aliases
    d = demand_mean if demand_mean is not None else daily_demand_mean
    d_std = demand_std if demand_std != 0.0 else (daily_demand_std if daily_demand_std is not None else 0.0)
    L = lead_time if lead_time is not None else lead_time_days
    l_std = lead_time_std if lead_time_std != 0.0 else (lead_time_std_days if lead_time_std_days is not None else 0.0)

    # Validate service level
    if service_level is None:
        raise ValueError("Missing required numerical input: 'service_level' must be provided.")
    csl = float(service_level)
    if csl <= 0.0 or csl >= 1.0:
        raise ValueError(f"Service level must be strictly between 0 and 1, got {service_level}")

    # Validate direct LTD vs (d, L) inputs
    using_direct_ltd = (lead_time_demand_mean is not None and lead_time_demand_std is not None)
    if not using_direct_ltd:
        if d is None:
            raise ValueError("Missing required numerical input: 'demand_mean' (d) must be provided.")
        if L is None:
            raise ValueError("Missing required numerical input: 'lead_time' (L) must be provided.")

        if d < 0:
            raise ValueError(f"Demand mean cannot be negative, got {d}")
        if L < 0:
            raise ValueError(f"Lead time cannot be negative, got {L}")
        if d_std < 0:
            raise ValueError(f"Demand standard deviation cannot be negative, got {d_std}")
        if l_std < 0:
            raise ValueError(f"Lead time standard deviation cannot be negative, got {l_std}")

        d = float(d)
        L = float(L)
        d_std = float(d_std)
        l_std = float(l_std)
    else:
        ltd_m = float(lead_time_demand_mean)
        ltd_s = float(lead_time_demand_std)
        if ltd_m < 0 or ltd_s < 0:
            raise ValueError("Lead time demand mean and std dev cannot be negative")

    # 2. Determine Case and Compute sigma_LT, SS, and ROP
    z = standard_normal_ppf(csl)

    if using_direct_ltd:
        scenario = "direct lead time demand distribution"
        mu_ltd = float(lead_time_demand_mean)
        sigma_lt = float(lead_time_demand_std)
        ss = z * sigma_lt
        rop = mu_ltd + ss
        assumptions = [
            "Lead time demand is directly specified as a normal distribution.",
            f"Target cycle service level CSL = {csl * 100:.2f}% (z = {z:.4f}).",
            "Continuous review policy (s, Q): inventory position is monitored continuously."
        ]
    elif d_std == 0.0 and l_std == 0.0:
        # Case 1: constant demand and constant lead time
        scenario = "constant demand and constant lead time"
        mu_ltd = d * L
        sigma_lt = 0.0
        ss = 0.0
        rop = d * L + ss
        assumptions = [
            f"Demand rate is constant and deterministic at {d} units per period (sigma_d = 0).",
            f"Lead time is constant and deterministic at {L} periods (sigma_L = 0).",
            "Zero safety stock is required due to complete certainty in demand and lead time.",
            "Continuous review policy (s, Q)."
        ]
    elif d_std > 0.0 and l_std == 0.0:
        # Case 2: variable demand and constant lead time
        scenario = "variable demand and constant lead time"
        mu_ltd = d * L
        sigma_lt = d_std * math.sqrt(L)
        ss = z * sigma_lt
        rop = d * L + ss
        assumptions = [
            f"Demand is variable and normally distributed with mean {d} and std dev {d_std} per period.",
            f"Lead time is constant and deterministic at {L} periods (sigma_L = 0).",
            "Lead time demand standard deviation scales with the square root of lead time: sigma_LT = sigma_d * sqrt(L).",
            f"Target cycle service level CSL = {csl * 100:.2f}% (z = {z:.4f}).",
            "Continuous review policy (s, Q)."
        ]
    elif d_std == 0.0 and l_std > 0.0:
        # Case 3: constant demand and variable lead time
        scenario = "constant demand and variable lead time"
        mu_ltd = d * L
        sigma_lt = d * l_std
        ss = z * sigma_lt
        rop = d * L + ss
        assumptions = [
            f"Demand rate is constant and fixed at {d} units per period (sigma_d = 0).",
            f"Lead time is variable and normally distributed with mean {L} and std dev {l_std} periods.",
            "Lead time demand standard deviation is proportional to demand: sigma_LT = d * sigma_L.",
            f"Target cycle service level CSL = {csl * 100:.2f}% (z = {z:.4f}).",
            "Continuous review policy (s, Q)."
        ]
    else:
        # Case 4: variable demand and variable lead time
        # Formula: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        scenario = "variable demand and variable lead time"
        mu_ltd = d * L
        sigma_lt = math.sqrt(L * (d_std ** 2) + (d ** 2) * (l_std ** 2))
        ss = z * sigma_lt
        rop = d * L + ss
        assumptions = [
            f"Demand is variable with mean {d} and std dev {d_std} per period.",
            f"Lead time is variable with mean {L} and std dev {l_std} periods.",
            "Demand per period and lead time duration are mutually independent normal random variables.",
            "Combined uncertainty follows: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2).",
            f"Target cycle service level CSL = {csl * 100:.2f}% (z = {z:.4f}).",
            "Continuous review policy (s, Q)."
        ]

    # Additional metrics
    stockout_risk = 1.0 - csl
    loss_z = unit_normal_loss(z) if sigma_lt > 0 else 0.0
    expected_shortage_per_cycle = sigma_lt * loss_z

    ss_holding_cost = None
    if holding_cost_per_unit is not None:
        if holding_cost_per_unit < 0:
            raise ValueError(f"Holding cost cannot be negative, got {holding_cost_per_unit}")
        ss_holding_cost = ss * float(holding_cost_per_unit)

    # 3. Structured Output
    inputs_dict = {
        "service_level": csl,
        "demand_mean": d,
        "demand_std": d_std,
        "lead_time": L,
        "lead_time_std": l_std,
        "holding_cost_per_unit": holding_cost_per_unit,
        "lead_time_demand_mean": lead_time_demand_mean,
        "lead_time_demand_std": lead_time_demand_std,
    }

    intermediates_dict = {
        "configuration_scenario": scenario,
        "z_score": round(z, 4),
        "expected_lead_time_demand_mu_LTD": round(mu_ltd, 2),
        "lead_time_demand_std_sigma_LT": round(sigma_lt, 2),
        "safety_stock_SS": round(ss, 2),
        "reorder_point_ROP": round(rop, 2),
        "stockout_risk_per_cycle": round(stockout_risk, 4),
        "expected_shortage_per_cycle": round(expected_shortage_per_cycle, 4),
        "annual_holding_cost_ss": round(ss_holding_cost, 2) if ss_holding_cost is not None else None,
    }

    final_answer_dict = {
        "reorder_point": round(rop, 2),
        "safety_stock": round(ss, 2),
        "service_level": round(csl, 4),
        "z_score": round(z, 4),
        "expected_lead_time_demand": round(mu_ltd, 2),
    }

    res = InventoryResult({
        "tool": "calculate_rop",
        "inputs": inputs_dict,
        "intermediate_calculations": intermediates_dict,
        "final_answer": final_answer_dict,
        "assumptions": assumptions,
        # Backward-compatible convenience keys:
        "service_level": round(csl, 4),
        "z_score": round(z, 4),
        "lead_time_demand_mean": round(mu_ltd, 2),
        "lead_time_demand_std": round(sigma_lt, 2),
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "stockout_risk": round(stockout_risk, 4),
        "lead_time_type": scenario,
        "annual_holding_cost_ss": round(ss_holding_cost, 2) if ss_holding_cost is not None else None,
        "daily_demand_mean": d,
        "daily_demand_std": d_std,
        "lead_time_mean_days": L,
        "lead_time_std_days": l_std,
    })
    return res


# Backward-compatible function alias
calculate_safety_stock_rop = calculate_rop


# ===========================================================================
# 4. Fixed-Period Review Model (Periodic Review / Order-Up-To)
# ===========================================================================

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    service_level: Optional[float] = 0.95,
    inventory_position: Optional[float] = None,
    current_on_hand: Optional[float] = None,
    current_on_order: float = 0.0,
    backorders: float = 0.0,
    safety_stock: Optional[float] = None,
) -> InventoryResult:
    """
    Calculate Fixed-Period Review (Periodic Review / Order-Up-To) Policy:
    Protection period = review period + lead time
    Target inventory = demand during protection period + safety stock
    Order quantity = max(0, target inventory - inventory position)

    Validates inputs, executes intermediate calculations, and returns structured result with assumptions.
    Never invents missing numerical inputs.
    """
    # 1. Input Validation
    if review_period is None:
        raise ValueError("Missing required numerical input: 'review_period' (T or P) must be provided.")
    if lead_time is None:
        raise ValueError("Missing required numerical input: 'lead_time' (L) must be provided.")
    if demand_mean is None:
        raise ValueError("Missing required numerical input: 'demand_mean' (d) must be provided.")
    if service_level is None and safety_stock is None:
        raise ValueError("Missing required numerical input: 'service_level' (CSL) or explicit 'safety_stock' must be provided.")

    if not isinstance(review_period, (int, float)) or review_period <= 0:
        raise ValueError(f"Review period T must be > 0, got {review_period}")
    if not isinstance(lead_time, (int, float)) or lead_time < 0:
        raise ValueError(f"Lead time L cannot be negative, got {lead_time}")
    if not isinstance(demand_mean, (int, float)) or demand_mean < 0:
        raise ValueError(f"Demand mean cannot be negative, got {demand_mean}")
    if not isinstance(demand_std, (int, float)) or demand_std < 0:
        raise ValueError(f"Demand std cannot be negative, got {demand_std}")

    if service_level is not None:
        csl = float(service_level)
        if csl <= 0.0 or csl >= 1.0:
            raise ValueError(f"Service level must be strictly in (0, 1), got {service_level}")
    else:
        csl = None

    P = float(review_period)
    L = float(lead_time)
    d = float(demand_mean)
    sigma_d = float(demand_std)

    # 2. Mathematical Calculations
    # Protection period = review period + lead time
    protection_period = P + L

    # Demand during protection period
    expected_demand_protection = d * protection_period
    std_demand_protection = sigma_d * math.sqrt(protection_period)

    # Safety stock
    if safety_stock is not None:
        calc_ss = float(safety_stock)
        z = calc_ss / std_demand_protection if std_demand_protection > 0 else 0.0
        if csl is None and std_demand_protection > 0:
            csl = standard_normal_cdf(z)
    else:
        z = standard_normal_ppf(csl)
        calc_ss = z * std_demand_protection

    # Target inventory = demand during protection period + safety stock
    target_inventory = expected_demand_protection + calc_ss

    # Determine inventory position
    # If explicitly passed, use it; else if current_on_hand passed, derive it
    inv_pos = None
    order_quantity = None
    if inventory_position is not None:
        inv_pos = float(inventory_position)
        order_quantity = max(0.0, target_inventory - inv_pos)
    elif current_on_hand is not None:
        on_hand = float(current_on_hand)
        on_order = float(current_on_order) if current_on_order is not None else 0.0
        bo = float(backorders) if backorders is not None else 0.0
        inv_pos = on_hand + on_order - bo
        order_quantity = max(0.0, target_inventory - inv_pos)

    # 3. Model Assumptions
    csl_str = f"Target cycle service level CSL = {csl * 100:.2f}% (z = {z:.4f})" if csl is not None else f"Safety stock fixed at {calc_ss} units"
    assumptions = [
        f"Inventory is reviewed periodically at fixed intervals of T = {P} periods.",
        f"Protection period covers review period + lead time: T + L = {protection_period} periods.",
        "Demand per period is independent and identically distributed with constant mean and variance.",
        f"{csl_str} over the protection interval.",
        "Order quantity brings the current inventory position up to the target inventory level: max(0, Target - IP).",
        "If inventory position already meets or exceeds target inventory, order quantity is 0."
    ]

    # 4. Structured Output
    inputs_dict = {
        "review_period": P,
        "lead_time": L,
        "demand_mean": d,
        "demand_std": sigma_d,
        "service_level": csl,
        "inventory_position": inv_pos,
        "current_on_hand": current_on_hand,
        "current_on_order": current_on_order,
        "backorders": backorders,
        "safety_stock": calc_ss,
    }

    intermediates_dict = {
        "protection_period": round(protection_period, 2),
        "expected_demand_during_protection": round(expected_demand_protection, 2),
        "std_dev_during_protection": round(std_demand_protection, 2),
        "z_score": round(z, 4),
        "safety_stock": round(calc_ss, 2),
        "target_inventory": round(target_inventory, 2),
        "inventory_position": round(inv_pos, 2) if inv_pos is not None else None,
        "order_quantity": round(order_quantity, 2) if order_quantity is not None else None,
    }

    final_answer_dict = {
        "target_inventory": round(target_inventory, 2),
        "protection_period": round(protection_period, 2),
        "safety_stock": round(calc_ss, 2),
        "recommended_order_quantity": round(order_quantity, 2) if order_quantity is not None else None,
        "inventory_position": round(inv_pos, 2) if inv_pos is not None else None,
    }

    res = InventoryResult({
        "tool": "calculate_periodic_review",
        "inputs": inputs_dict,
        "intermediate_calculations": intermediates_dict,
        "final_answer": final_answer_dict,
        "assumptions": assumptions,
        # Backward-compatible convenience keys:
        "review_period_T": round(P, 2),
        "lead_time_L": round(L, 2),
        "protection_interval": round(protection_period, 2),
        "demand_mean_per_period": round(d, 2),
        "demand_std_per_period": round(sigma_d, 2),
        "service_level": round(csl, 4) if csl is not None else None,
        "z_score": round(z, 4),
        "expected_demand_protection_interval": round(expected_demand_protection, 2),
        "std_demand_protection_interval": round(std_demand_protection, 2),
        "safety_stock": round(calc_ss, 2),
        "order_up_to_level_S": round(target_inventory, 2),
        "target_inventory_level": round(target_inventory, 2),
        "inventory_on_hand": round(current_on_hand, 2) if current_on_hand is not None else 0.0,
        "inventory_on_order": round(current_on_order, 2),
        "backorders": round(backorders, 2),
        "current_inventory_position": round(inv_pos, 2) if inv_pos is not None else 0.0,
        "recommended_order_quantity": round(order_quantity, 2) if order_quantity is not None else 0.0,
    })
    return res


# Backward-compatible function alias
calculate_fixed_period_review = calculate_periodic_review


# ===========================================================================
# 5. Newsvendor Model (Single-Period Stochastic Demand)
# ===========================================================================

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    purchase_cost: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    salvage_value: float = 0.0,
    goodwill_cost: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> InventoryResult:
    """
    Calculate Newsvendor Model parameters:
    - Underage cost (Cu)
    - Overage cost (Co)
    - Critical ratio (CR = Cu / (Cu + Co))
    - Recommended order quantity (Q* = mu + z* * sigma)

    Validates inputs, executes intermediate calculations, and returns structured result with assumptions.
    Never invents missing numerical inputs.
    """
    # 1. Input Validation
    has_direct_cu_co = (underage_cost is not None and overage_cost is not None)

    if not has_direct_cu_co:
        if selling_price is None:
            raise ValueError("Missing required numerical input: 'selling_price' must be provided (or provide 'underage_cost' and 'overage_cost').")
        if purchase_cost is None:
            raise ValueError("Missing required numerical input: 'purchase_cost' must be provided (or provide 'underage_cost' and 'overage_cost').")

    if demand_mean is None:
        raise ValueError("Missing required numerical input: 'demand_mean' must be provided.")
    if demand_std is None:
        raise ValueError("Missing required numerical input: 'demand_std' must be provided.")

    if not isinstance(demand_mean, (int, float)) or demand_mean <= 0:
        raise ValueError(f"Demand mean must be > 0, got {demand_mean}")
    if not isinstance(demand_std, (int, float)) or demand_std < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {demand_std}")

    s = float(salvage_value)
    g = float(goodwill_cost)
    mu = float(demand_mean)
    sigma = float(demand_std)

    if has_direct_cu_co:
        cu = float(underage_cost)
        co = float(overage_cost)
        p = float(selling_price) if selling_price is not None else None
        c = float(purchase_cost) if purchase_cost is not None else None
    else:
        if not isinstance(selling_price, (int, float)) or selling_price <= 0:
            raise ValueError(f"Selling price must be > 0, got {selling_price}")
        if not isinstance(purchase_cost, (int, float)) or purchase_cost <= 0:
            raise ValueError(f"Purchase cost must be > 0, got {purchase_cost}")
        if selling_price < purchase_cost:
            raise ValueError(f"Selling price (${selling_price}) cannot be less than purchase cost (${purchase_cost})")
        if salvage_value > purchase_cost:
            raise ValueError(f"Salvage value (${salvage_value}) cannot exceed purchase cost (${purchase_cost})")

        p = float(selling_price)
        c = float(purchase_cost)
        cu = p - c + g
        co = c - s

    if cu <= 0:
        raise ValueError(f"Cost of underage Cu must be > 0, got {cu}")
    if co <= 0:
        raise ValueError(f"Cost of overage Co must be > 0, got {co}")

    # 2. Mathematical Calculations
    # Critical ratio: CR = Cu / (Cu + Co)
    cr = cu / (cu + co)

    if sigma > 0:
        z_star = standard_normal_ppf(cr)
        q_star = mu + (z_star * sigma)
        q_star = max(0.0, q_star)
        effective_z = (q_star - mu) / sigma

        unit_loss = unit_normal_loss(effective_z)
        exp_lost_sales = sigma * unit_loss
        exp_leftover = q_star - mu + exp_lost_sales
        exp_sales = mu - exp_lost_sales
    else:
        z_star = 0.0
        q_star = mu
        effective_z = 0.0
        exp_lost_sales = 0.0
        exp_leftover = 0.0
        exp_sales = mu

    in_stock_prob = standard_normal_cdf(effective_z) if sigma > 0 else 1.0

    if p is not None and c is not None:
        exp_rev = p * exp_sales
        exp_cost = c * q_star
        exp_salvage = s * exp_leftover
        exp_penalty = g * exp_lost_sales
        exp_profit = exp_rev + exp_salvage - exp_cost - exp_penalty
        pi_profit = (p - c) * mu
        evpi = max(0.0, pi_profit - exp_profit)
    else:
        exp_rev = None
        exp_cost = None
        exp_salvage = None
        exp_penalty = None
        exp_profit = None
        pi_profit = None
        evpi = None

    # 3. Model Assumptions
    cu_desc = f"Cost of underage Cu = ${cu:.2f}/unit" if has_direct_cu_co else f"Cost of underage Cu = p - c + g = ${cu:.2f}/unit (profit margin plus stockout penalty)."
    co_desc = f"Cost of overage Co = ${co:.2f}/unit" if has_direct_cu_co else f"Cost of overage Co = c - s = ${co:.2f}/unit (loss per unit unsold)."
    assumptions = [
        "Single sales period for perishable, seasonal, or short-lifecycle items.",
        "A single replenishment order must be placed before demand is realized.",
        "Demand follows a normal distribution with known mean mu and std dev sigma.",
        cu_desc,
        co_desc,
        f"Critical ratio CR = Cu / (Cu + Co) = {cr:.4f} defines the optimal in-stock probability.",
    ]
    if not has_direct_cu_co:
        assumptions.append("Unsold units at period end are salvaged at unit salvage value s with no disposal fee.")

    # 4. Structured Output
    inputs_dict = {
        "selling_price": p,
        "purchase_cost": c,
        "demand_mean": mu,
        "demand_std": sigma,
        "salvage_value": s,
        "goodwill_cost": g,
        "underage_cost": cu,
        "overage_cost": co,
    }

    intermediates_dict = {
        "cost_of_underage_Cu": round(cu, 2),
        "cost_of_overage_Co": round(co, 2),
        "critical_ratio": round(cr, 4),
        "z_score": round(effective_z, 4),
        "expected_lost_sales": round(exp_lost_sales, 2),
        "expected_leftover_units": round(exp_leftover, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_revenue": round(exp_rev, 2) if exp_rev is not None else None,
        "expected_purchase_cost": round(exp_cost, 2) if exp_cost is not None else None,
        "expected_salvage_revenue": round(exp_salvage, 2) if exp_salvage is not None else None,
        "expected_goodwill_penalty": round(exp_penalty, 2) if exp_penalty is not None else None,
        "expected_profit": round(exp_profit, 2) if exp_profit is not None else None,
        "expected_profit_perfect_information": round(pi_profit, 2) if pi_profit is not None else None,
        "expected_value_of_perfect_information_EVPI": round(evpi, 2) if evpi is not None else None,
        "in_stock_probability": round(in_stock_prob, 4),
    }

    final_answer_dict = {
        "recommended_order_quantity": round(q_star, 2),
        "critical_ratio": round(cr, 4),
        "cost_of_underage": round(cu, 2),
        "cost_of_overage": round(co, 2),
        "expected_profit": round(exp_profit, 2) if exp_profit is not None else None,
    }

    res = InventoryResult({
        "tool": "calculate_newsvendor",
        "inputs": inputs_dict,
        "intermediate_calculations": intermediates_dict,
        "final_answer": final_answer_dict,
        "assumptions": assumptions,
        # Backward-compatible convenience keys:
        "selling_price": round(p, 2) if p is not None else None,
        "purchase_cost": round(c, 2) if c is not None else None,
        "salvage_value": round(s, 2),
        "goodwill_cost": round(g, 2),
        "cost_underage_Cu": round(cu, 2),
        "cost_overage_Co": round(co, 2),
        "critical_fractile": round(cr, 4),
        "demand_mean": round(mu, 2),
        "demand_std": round(sigma, 2),
        "z_score": round(effective_z, 4),
        "optimal_order_quantity": round(q_star, 2),
        "recommended_order_quantity": round(q_star, 2),
        "expected_lost_sales": round(exp_lost_sales, 2),
        "expected_leftover_units": round(exp_leftover, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_revenue": round(exp_rev, 2) if exp_rev is not None else None,
        "expected_purchase_cost": round(exp_cost, 2) if exp_cost is not None else None,
        "expected_salvage_revenue": round(exp_salvage, 2) if exp_salvage is not None else None,
        "expected_goodwill_penalty": round(exp_penalty, 2) if exp_penalty is not None else None,
        "expected_profit": round(exp_profit, 2) if exp_profit is not None else None,
        "expected_profit_under_perfect_information": round(pi_profit, 2) if pi_profit is not None else None,
        "expected_value_of_perfect_information_EVPI": round(evpi, 2) if evpi is not None else None,
        "in_stock_probability": round(in_stock_prob, 4),
    })
    return res
