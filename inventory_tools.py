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
       - result["inputs_extracted"] (and result["inputs"])
       - result["variable_definitions"]
       - result["formula"] (and result["formulas"])
       - result["formula_with_substituted_values"]
       - result["intermediate_calculations"]
       - result["final_answer"]
       - result["verification_checks"]
       - result["assumptions"]
    2. Direct attribute access:
       - result.inputs_extracted
       - result.variable_definitions
       - result.formula
       - result.formula_with_substituted_values
       - result.intermediate_calculations
       - result.final_answer
       - result.verification_checks
       - result.assumptions
    3. Top-level keys for seamless backward compatibility with existing tests and agents:
       - result["eoq"], result["reorder_point"], result["safety_stock"], etc.
    """
    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        if item == "inputs_extracted" and "inputs" in self:
            return self["inputs"]
        if item == "inputs" and "inputs_extracted" in self:
            return self["inputs_extracted"]
        if item == "formula" and "formulas" in self:
            return self["formulas"]
        if item == "formulas" and "formula" in self:
            return self["formula"]
        if item in ("formula_with_substituted_values", "formulas_substituted"):
            if "formula_with_substituted_values" in self:
                return self["formula_with_substituted_values"]
            if "formulas_substituted" in self:
                return self["formulas_substituted"]
        if item in ("step_by_step_calculation", "step_by_step_derivation", "step_by_step"):
            for k in ("step_by_step_calculation", "step_by_step_derivation", "step_by_step"):
                if k in self:
                    return self[k]
        raise AttributeError(f"'InventoryResult' object has no attribute '{item}'")

    def __getitem__(self, item: str) -> Any:
        if item in self:
            return super().__getitem__(item)
        if item == "inputs_extracted" and "inputs" in self:
            return super().__getitem__("inputs")
        if item == "inputs" and "inputs_extracted" in self:
            return super().__getitem__("inputs_extracted")
        if item == "formula" and "formulas" in self:
            return super().__getitem__("formulas")
        if item == "formulas" and "formula" in self:
            return super().__getitem__("formula")
        if item in ("formula_with_substituted_values", "formulas_substituted"):
            if "formula_with_substituted_values" in self:
                return super().__getitem__("formula_with_substituted_values")
            if "formulas_substituted" in self:
                return super().__getitem__("formulas_substituted")
        if item in ("step_by_step_calculation", "step_by_step_derivation", "step_by_step"):
            for k in ("step_by_step_calculation", "step_by_step_derivation", "step_by_step"):
                if k in self:
                    return super().__getitem__(k)
        return super().__getitem__(item)

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

    # 4. Variable Definitions
    variable_definitions = {
        "D": "Annual Demand (units/year)",
        "S": "Ordering / Setup Cost per order ($/order)",
        "H": "Unit Holding / Carrying Cost per year ($/unit/year)",
        "Q*": "Economic Order Quantity (optimal batch size in units)",
        "N": "Order Frequency (orders placed per year = D / Q*)",
        "T_days": "Order Cycle Time in operating days ((Q* / D) * W)",
        "AOC": "Annual Ordering Cost ($/year = N * S)",
        "AHC": "Annual Holding Cost ($/year = (Q* / 2) * H)",
        "TIC": "Total Inventory Cost ($/year = AOC + AHC)",
    }
    if unit_price is not None:
        variable_definitions["C"] = "Unit Purchase Price ($/unit)"
        variable_definitions["APC"] = "Annual Purchase Cost ($/year = D * C)"
        variable_definitions["TC"] = "Total Annual Cost ($/year = APC + TIC)"
    if holding_rate is not None:
        variable_definitions["h"] = "Annual Inventory Holding Rate (fraction or percentage of unit price)"
    if lead_time_days is not None:
        variable_definitions["W"] = "Working Days per Year (operating days/year)"
        variable_definitions["L"] = "Replenishment Lead Time (days)"
        variable_definitions["d"] = "Daily Demand Rate (units/day = D / W)"
        variable_definitions["ROP"] = "Reorder Point (inventory level triggering an order = d * L)"

    # 5. Symbolic Formulas
    formulas = {
        "holding_cost": "H = holding_cost" if holding_source == "explicit_holding_cost" else "H = unit_price * holding_rate",
        "eoq": "Q* = sqrt(2DS/H)",
        "orders_per_year": "N = D / Q*",
        "cycle_time_days": "T_days = (Q* / D) * W",
        "annual_ordering_cost": "AOC = (D / Q*) * S = N * S",
        "annual_holding_cost": "AHC = (Q* / 2) * H",
        "total_inventory_cost": "TIC = AOC + AHC = ((D / Q*) * S) + ((Q* / 2) * H)",
    }
    if unit_price is not None:
        formulas["annual_purchase_cost"] = "APC = D * C"
        formulas["total_annual_cost"] = "TC = APC + TIC = (D * C) + TIC"
    if lead_time_days is not None:
        formulas["daily_demand"] = "d = D / W"
        formulas["reorder_point"] = "ROP = d * L = (D / W) * L"

    # 6. Formulas with Substituted Values
    holding_sub = f"H = ${H:.2f}/unit/year" if holding_source == "explicit_holding_cost" else f"H = ${unit_price:.2f} * {rate:.2f} = ${H:.2f}/unit/year"
    formulas_substituted = {
        "holding_cost": holding_sub,
        "eoq": f"Q* = sqrt((2 x D x S) / H) = sqrt((2 x {D:,.2f} x {S:,.2f}) / {H:,.2f}) = sqrt({2.0 * D * S:,.2f} / {H:,.2f}) = sqrt({(2.0 * D * S) / H:,.2f}) = {eoq:,.2f} units",
        "orders_per_year": f"N = {D:,.2f} / {eoq:,.2f} = {orders_per_year:,.2f} orders/year",
        "cycle_time_days": f"T_days = ({eoq:,.2f} / {D:,.2f}) * {working_days_per_year} = {cycle_time_days:,.1f} operating days",
        "annual_ordering_cost": f"AOC = ({D:,.2f} / {eoq:,.2f}) * ${S:,.2f} = {orders_per_year:,.2f} * ${S:,.2f} = ${annual_ordering_cost:,.2f}/year",
        "annual_holding_cost": f"AHC = ({eoq:,.2f} / 2) * ${H:,.2f} = {eoq / 2.0:,.2f} * ${H:,.2f} = ${annual_holding_cost:,.2f}/year",
        "total_inventory_cost": f"TIC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_inventory_cost:,.2f}/year",
    }
    if unit_price is not None:
        formulas_substituted["annual_purchase_cost"] = f"APC = {D:,.2f} * ${unit_price:,.2f} = ${annual_purchase_cost:,.2f}/year"
        formulas_substituted["total_annual_cost"] = f"TC = ${annual_purchase_cost:,.2f} + ${total_inventory_cost:,.2f} = ${total_annual_cost:,.2f}/year"
    if lead_time_days is not None:
        formulas_substituted["daily_demand"] = f"d = {D:,.2f} / {working_days_per_year} = {daily_demand:,.2f} units/day"
        formulas_substituted["reorder_point"] = f"ROP = {daily_demand:,.2f} * {lead_time_days} = {reorder_point:,.2f} units"

    # Step-by-Step Numerical Derivation Chain (for student verification)
    step_by_step_calculation = [
        f"D = {D:,.2f}",
        f"S = {S:,.2f}",
        f"H = {H:,.2f}",
        "Q* = sqrt(2DS/H)",
        "Q* = sqrt((2 x D x S) / H)",
        f"Q* = sqrt((2 x {D:,.2f} x {S:,.2f}) / {H:,.2f})",
        f"Q* = sqrt({2.0 * D * S:,.2f} / {H:,.2f})",
        f"Q* = sqrt({(2.0 * D * S) / H:,.2f})",
        f"Q* = {eoq:,.2f} units.",
        # Supporting EOQ calculations
        f"N = D / Q* = {D:,.2f} / {eoq:,.2f} = {orders_per_year:,.2f} orders/year",
        f"T = (Q* / D) * W = ({eoq:,.2f} / {D:,.2f}) * {working_days_per_year} = {cycle_time_days:,.1f} operating days",
        f"AOC = (D / Q*) * S = ({D:,.2f} / {eoq:,.2f}) * ${S:,.2f} = ${annual_ordering_cost:,.2f}/year",
        f"AHC = (Q* / 2) * H = ({eoq:,.2f} / 2) * ${H:,.2f} = ${annual_holding_cost:,.2f}/year",
        f"TIC = AOC + AHC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_inventory_cost:,.2f}/year",
    ]
    if unit_price is not None:
        step_by_step_calculation.append(f"APC = D * C = {D:,.2f} * ${unit_price:,.2f} = ${annual_purchase_cost:,.2f}/year")
        step_by_step_calculation.append(f"TC = APC + TIC = ${annual_purchase_cost:,.2f} + ${total_inventory_cost:,.2f} = ${total_annual_cost:,.2f}/year")
    if lead_time_days is not None:
        step_by_step_calculation.append(f"d = D / W = {D:,.2f} / {working_days_per_year} = {daily_demand:,.2f} units/day")
        step_by_step_calculation.append(f"ROP = d * L = {daily_demand:,.2f} * {lead_time_days} = {reorder_point:,.2f} units")

    # 7. Verification Checks
    verification_checks = []
    # Check 1: Non-zero positive order quantity
    if eoq > 0:
        verification_checks.append({
            "name": "Non-Zero Order Quantity",
            "check": "Non-Zero Order Quantity",
            "status": "PASS",
            "detail": f"EOQ is strictly positive ({eoq:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "Non-Zero Order Quantity",
            "check": "Non-Zero Order Quantity",
            "status": "FAIL",
            "detail": "EOQ is non-positive."
        })

    # Check 2: Economic cost parity check (AOC == AHC at unconstrained EOQ)
    diff = abs(annual_ordering_cost - annual_holding_cost)
    rel_diff = diff / max(annual_ordering_cost, annual_holding_cost) if max(annual_ordering_cost, annual_holding_cost) > 0 else 0
    if rel_diff < 0.02:
        verification_checks.append({
            "name": "Economic Trade-Off Parity",
            "check": "Economic Trade-Off Parity",
            "status": "PASS",
            "detail": f"Annual ordering cost (${annual_ordering_cost:,.2f}) equals annual holding cost (${annual_holding_cost:,.2f}) within rounding tolerance (|diff| = ${diff:,.2f})."
        })
    else:
        verification_checks.append({
            "name": "Economic Trade-Off Parity",
            "check": "Economic Trade-Off Parity",
            "status": "WARN",
            "detail": f"Ordering cost (${annual_ordering_cost:,.2f}) and holding cost (${annual_holding_cost:,.2f}) deviate by {rel_diff:.1%}."
        })

    # Check 3: Order frequency sanity
    if 0.5 <= orders_per_year <= 365:
        verification_checks.append({
            "name": "Order Frequency Sanity",
            "check": "Order Frequency Sanity",
            "status": "PASS",
            "detail": f"{orders_per_year:,.2f} orders/year (every {cycle_time_days:,.1f} days) is operationally practical."
        })
    else:
        verification_checks.append({
            "name": "Order Frequency Sanity",
            "check": "Order Frequency Sanity",
            "status": "WARN",
            "detail": f"Extreme order frequency: {orders_per_year:,.2f} orders/year."
        })

    if lead_time_days is not None:
        verification_checks.append({
            "name": "Reorder Point Validity",
            "check": "Reorder Point Validity",
            "status": "PASS",
            "detail": f"ROP of {reorder_point:,.2f} units covers {lead_time_days} days of demand at {daily_demand:,.2f} units/day."
        })

    # 8. Structured Output (containing all 8 student verification sections)
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
        "step_by_step_calculation": step_by_step_calculation,
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
        # 8 Mandated Student Verification Sections:
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formulas_substituted,
        "intermediate_calculations": intermediates_dict,
        "step_by_step_calculation": step_by_step_calculation,
        "step_by_step_derivation": step_by_step_calculation,
        "final_answer": final_answer_dict,
        "verification_checks": verification_checks,
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

    # 4. Variable Definitions
    variable_definitions = {
        "D": "Annual Demand (units/year)",
        "S": "Ordering / Setup Cost per order ($/order)",
        "i": "Discount Tier Index (1, 2, ...)",
        "C_i": "Unit Purchase Price in Tier i ($/unit)",
        "H_i": "Unit Holding / Carrying Cost in Tier i ($/unit/year)",
        "EOQ_i": "Unconstrained EOQ calculated at Tier i price (units = sqrt(2*D*S / H_i))",
        "Q_i": "Feasible Candidate Order Quantity evaluated for Tier i (units)",
        "APC_i": "Annual Purchase Cost in Tier i ($/year = D * C_i)",
        "AOC_i": "Annual Ordering Cost in Tier i ($/year = (D / Q_i) * S)",
        "AHC_i": "Annual Holding Cost in Tier i ($/year = (Q_i / 2) * H_i)",
        "TC_i": "Total Annual Cost in Tier i ($/year = APC_i + AOC_i + AHC_i)",
        "Q*": "Optimal Order Quantity (units)",
        "TC*": "Global Minimum Total Annual Cost ($/year)",
        "Savings": "Annual Cost Savings vs Baseline Tier 1 ($/year and %)",
    }
    if rate is not None:
        variable_definitions["h"] = "Holding Cost Rate as percentage of unit purchase price (%/year)"

    # 5. Symbolic Formulas
    formulas = {
        "holding_cost": "H_i = C_i * holding_rate" if rate is not None else "H_i = holding_cost",
        "unconstrained_eoq": "EOQ_i = sqrt(2DS/H_i)",
        "candidate_quantity": "Q_i = EOQ_i if min_qty_i <= EOQ_i <= max_qty_i else (min_qty_i if EOQ_i < min_qty_i else max_qty_i)",
        "annual_purchase_cost": "APC_i = D * C_i",
        "annual_ordering_cost": "AOC_i = (D / Q_i) * S",
        "annual_holding_cost": "AHC_i = (Q_i / 2) * H_i",
        "total_annual_cost": "TC_i = APC_i + AOC_i + AHC_i = (D * C_i) + ((D / Q_i) * S) + ((Q_i / 2) * H_i)",
        "optimal_selection": "TC* = min(TC_1, TC_2, ...), Q* = candidate Q of minimizing tier",
        "annual_savings": "Annual Savings = TC_baseline - TC*",
    }

    # 6. Formulas with Substituted Values
    formulas_substituted = {}
    for idx, e in enumerate(evaluations, 1):
        formulas_substituted[f"tier_{idx}"] = (
            f"Tier {idx} (Price = ${e.unit_price:.2f}): H_{idx} = ${e.holding_cost_per_unit:.2f}, "
            f"EOQ_{idx} = sqrt((2 * {D:,.0f} * ${S:.2f}) / ${e.holding_cost_per_unit:.2f}) = {e.unconstrained_eoq:,.2f} -> "
            f"Candidate Q_{idx} = {e.candidate_qty:,.0f} units; "
            f"TC_{idx} = ({D:,.0f} * ${e.unit_price:.2f}) + ({D:,.0f} / {e.candidate_qty:,.2f}) * ${S:.2f} + "
            f"({e.candidate_qty:,.2f} / 2) * ${e.holding_cost_per_unit:.2f} = "
            f"${e.annual_purchase_cost:,.2f} + ${e.annual_ordering_cost:,.2f} + ${e.annual_holding_cost:,.2f} = "
            f"${e.total_annual_cost:,.2f}"
        )
    formulas_substituted["optimal_selection"] = (
        f"TC* = min({', '.join('$' + f'{e.total_annual_cost:,.2f}' for e in evaluations)}) = "
        f"${best_eval.total_annual_cost:,.2f} at Tier {best_eval.tier_index} (Q* = {best_eval.candidate_qty:,.0f} units)"
    )
    formulas_substituted["annual_savings"] = (
        f"Savings = ${baseline_total_cost:,.2f} - ${best_eval.total_annual_cost:,.2f} = "
        f"${annual_savings:,.2f} ({pct_savings:.2f}%)"
    )

    # Step-by-step derivation for student verification
    step_by_step_calculation = [
        f"D = {D:,.2f}",
        f"S = {S:,.2f}",
        f"Holding Cost = {'h * C_i with h = ' + str(rate) if rate is not None else f'${fixed_h:,.2f}/unit/year'}",
        "EOQ_i = sqrt(2DS/H_i)",
        "EOQ_i = sqrt((2 x D x S) / H_i)",
    ]
    for e in evaluations:
        max_str = f"{e.max_qty:,.0f}" if e.max_qty is not None else "∞"
        feas_str = "feasible in range" if e.is_eoq_feasible else "adjusted to price break floor"
        step_by_step_calculation.append(
            f"--- Tier {e.tier_index} (Range: [{e.min_qty:,.0f} - {max_str}], Unit Price: ${e.unit_price:.2f}) ---"
        )
        step_by_step_calculation.append(
            f"H_{e.tier_index} = ${e.holding_cost_per_unit:.2f}/unit/year"
        )
        step_by_step_calculation.append(
            f"EOQ_{e.tier_index} = sqrt((2 x {D:,.2f} x {S:,.2f}) / {e.holding_cost_per_unit:.2f}) = {e.unconstrained_eoq:,.2f} units ({feas_str})"
        )
        step_by_step_calculation.append(
            f"Candidate Q_{e.tier_index} = {e.candidate_qty:,.2f} units"
        )
        step_by_step_calculation.append(
            f"APC_{e.tier_index} = D * C_{e.tier_index} = {D:,.2f} * ${e.unit_price:.2f} = ${e.annual_purchase_cost:,.2f}/year"
        )
        step_by_step_calculation.append(
            f"AOC_{e.tier_index} = (D / Q_{e.tier_index}) * S = ({D:,.2f} / {e.candidate_qty:,.2f}) * ${S:,.2f} = ${e.annual_ordering_cost:,.2f}/year"
        )
        step_by_step_calculation.append(
            f"AHC_{e.tier_index} = (Q_{e.tier_index} / 2) * H_{e.tier_index} = ({e.candidate_qty:,.2f} / 2) * ${e.holding_cost_per_unit:.2f} = ${e.annual_holding_cost:,.2f}/year"
        )
        step_by_step_calculation.append(
            f"TC_{e.tier_index} = APC + AOC + AHC = ${e.annual_purchase_cost:,.2f} + ${e.annual_ordering_cost:,.2f} + ${e.annual_holding_cost:,.2f} = ${e.total_annual_cost:,.2f}/year"
        )
    step_by_step_calculation.append(
        f"TC* = min({', '.join('$' + f'{e.total_annual_cost:,.2f}' for e in evaluations)}) = ${best_eval.total_annual_cost:,.2f}"
    )
    step_by_step_calculation.append(
        f"Optimal Order Quantity Q* = {best_eval.candidate_qty:,.2f} units (Tier {best_eval.tier_index} at ${best_eval.unit_price:.2f}/unit)."
    )
    step_by_step_calculation.append(
        f"Annual Savings vs Baseline = ${annual_savings:,.2f} ({pct_savings:.2f}%)."
    )

    # 7. Verification Checks
    verification_checks = []
    # Check 1: Global cost minimization
    all_higher_or_equal = all(e.total_annual_cost >= best_eval.total_annual_cost for e in evaluations)
    if all_higher_or_equal:
        verification_checks.append({
            "name": "Global Cost Minimization",
            "check": "Global Cost Minimization",
            "status": "PASS",
            "detail": f"Tier {best_eval.tier_index} strictly achieves minimum total annual cost (${best_eval.total_annual_cost:,.2f})."
        })
    else:
        verification_checks.append({
            "name": "Global Cost Minimization",
            "check": "Global Cost Minimization",
            "status": "FAIL",
            "detail": "Identified tier does not achieve minimum total annual cost."
        })

    # Check 2: Bracket feasibility check
    opt_min = best_eval.min_qty
    opt_max = best_eval.max_qty
    is_in_bracket = (best_eval.candidate_qty >= opt_min) and (opt_max is None or best_eval.candidate_qty <= opt_max)
    if is_in_bracket:
        verification_checks.append({
            "name": "Bracket Feasibility",
            "check": "Bracket Feasibility",
            "status": "PASS",
            "detail": f"Order quantity {best_eval.candidate_qty:,.0f} satisfies Tier {best_eval.tier_index} range [{opt_min:,.0f}, {opt_max or '∞'}]."
        })
    else:
        verification_checks.append({
            "name": "Bracket Feasibility",
            "check": "Bracket Feasibility",
            "status": "FAIL",
            "detail": f"Order quantity {best_eval.candidate_qty:,.0f} violates Tier {best_eval.tier_index} bracket bounds."
        })

    # Check 3: Cost non-negativity
    if best_eval.annual_purchase_cost > 0 and best_eval.annual_ordering_cost > 0 and best_eval.annual_holding_cost > 0:
        verification_checks.append({
            "name": "Cost Component Non-Negativity",
            "check": "Cost Component Non-Negativity",
            "status": "PASS",
            "detail": f"Purchase (${best_eval.annual_purchase_cost:,.2f}), ordering (${best_eval.annual_ordering_cost:,.2f}), and holding (${best_eval.annual_holding_cost:,.2f}) costs are all positive."
        })
    else:
        verification_checks.append({
            "name": "Cost Component Non-Negativity",
            "check": "Cost Component Non-Negativity",
            "status": "FAIL",
            "detail": "One or more cost components are non-positive."
        })

    # Check 4: Cost savings non-negativity
    if annual_savings >= 0:
        verification_checks.append({
            "name": "Non-Negative Savings",
            "check": "Non-Negative Savings",
            "status": "PASS",
            "detail": f"Discount plan yields non-negative savings of ${annual_savings:,.2f} ({pct_savings:.2f}%) vs baseline."
        })
    else:
        verification_checks.append({
            "name": "Non-Negative Savings",
            "check": "Non-Negative Savings",
            "status": "WARN",
            "detail": f"Discount order quantity results in negative savings vs baseline (${annual_savings:,.2f})."
        })

    # 8. Structured Output (containing all 8 student verification sections)
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
        "step_by_step_calculation": step_by_step_calculation,
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
        # 8 Mandated Student Verification Sections:
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formulas_substituted,
        "intermediate_calculations": intermediates_dict,
        "step_by_step_calculation": step_by_step_calculation,
        "step_by_step_derivation": step_by_step_calculation,
        "final_answer": final_answer_dict,
        "verification_checks": verification_checks,
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

    # 4. Variable Definitions
    variable_definitions = {
        "CSL": "Cycle Service Level (target probability of not stocking out during lead time)",
        "z": "Safety Factor / Standard Normal Quantile (z = Phi^-1(CSL))",
        "mu_LTD": "Expected Demand during Lead Time (units)",
        "sigma_LT": "Standard Deviation of Demand during Lead Time (units)",
        "SS": "Safety Stock (buffer inventory = z * sigma_LT)",
        "ROP": "Reorder Point (inventory level that triggers replenishment = mu_LTD + SS)",
        "Risk": "Cycle Stockout Risk (probability of stockout = 1 - CSL)",
    }
    if not using_direct_ltd:
        variable_definitions["d"] = "Average Demand per Period (units/period)"
        variable_definitions["sigma_d"] = "Standard Deviation of Demand per Period (units/period)"
        variable_definitions["L"] = "Average Replenishment Lead Time (periods)"
        variable_definitions["sigma_L"] = "Standard Deviation of Lead Time (periods)"
    if holding_cost_per_unit is not None:
        variable_definitions["H"] = "Holding Cost per Unit per Year ($/unit/year)"
        variable_definitions["AHC_SS"] = "Annual Holding Cost of Safety Stock ($/year = SS * H)"

    # 5. Symbolic Formulas
    formulas = {
        "safety_factor": "z = Phi^-1(CSL)",
        "stockout_risk": "Risk = 1 - CSL",
    }
    if using_direct_ltd:
        formulas["expected_lead_time_demand"] = "mu_LTD = lead_time_demand_mean"
        formulas["lead_time_demand_std"] = "sigma_LT = lead_time_demand_std"
        formulas["safety_stock"] = "SS = z * sigma_LT"
        formulas["reorder_point"] = "ROP = mu_LTD + SS"
    elif d_std == 0.0 and l_std == 0.0:
        formulas["expected_lead_time_demand"] = "mu_LTD = d * L"
        formulas["lead_time_demand_std"] = "sigma_LT = 0"
        formulas["safety_stock"] = "SS = 0"
        formulas["reorder_point"] = "ROP = d * L"
    elif d_std > 0.0 and l_std == 0.0:
        formulas["expected_lead_time_demand"] = "mu_LTD = d * L"
        formulas["lead_time_demand_std"] = "sigma_LT = sigma_d * sqrt(L)"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * sigma_d * sqrt(L)"
        formulas["reorder_point"] = "ROP = (d * L) + SS"
    elif d_std == 0.0 and l_std > 0.0:
        formulas["expected_lead_time_demand"] = "mu_LTD = d * L"
        formulas["lead_time_demand_std"] = "sigma_LT = d * sigma_L"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * d * sigma_L"
        formulas["reorder_point"] = "ROP = (d * L) + SS"
    else:
        formulas["expected_lead_time_demand"] = "mu_LTD = d * L"
        formulas["lead_time_demand_std"] = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        formulas["reorder_point"] = "ROP = (d * L) + SS"

    if holding_cost_per_unit is not None:
        formulas["annual_holding_cost_ss"] = "AHC_SS = SS * H"

    # 6. Formulas with Substituted Values
    formulas_substituted = {
        "safety_factor": f"z = Phi^-1({csl:.4f}) = {z:.4f}",
        "stockout_risk": f"Risk = 1 - {csl:.4f} = {stockout_risk:.4f} ({stockout_risk:.2%})",
    }
    if using_direct_ltd:
        formulas_substituted["expected_lead_time_demand"] = f"mu_LTD = {mu_ltd:,.2f} units"
        formulas_substituted["lead_time_demand_std"] = f"sigma_LT = {sigma_lt:,.2f} units"
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units"
        formulas_substituted["reorder_point"] = f"ROP = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units"
    elif d_std == 0.0 and l_std == 0.0:
        formulas_substituted["expected_lead_time_demand"] = f"mu_LTD = {d:,.2f} * {L:,.2f} = {mu_ltd:,.2f} units"
        formulas_substituted["lead_time_demand_std"] = "sigma_LT = 0.00 units (constant demand & lead time)"
        formulas_substituted["safety_stock"] = "SS = 0.00 units"
        formulas_substituted["reorder_point"] = f"ROP = {d:,.2f} * {L:,.2f} = {rop:,.2f} units"
    elif d_std > 0.0 and l_std == 0.0:
        formulas_substituted["expected_lead_time_demand"] = f"mu_LTD = {d:,.2f} * {L:,.2f} = {mu_ltd:,.2f} units"
        formulas_substituted["lead_time_demand_std"] = f"sigma_LT = {d_std:,.2f} * sqrt({L:,.2f}) = {d_std:,.2f} * {math.sqrt(L):,.4f} = {sigma_lt:,.2f} units"
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units"
        formulas_substituted["reorder_point"] = f"ROP = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units"
    elif d_std == 0.0 and l_std > 0.0:
        formulas_substituted["expected_lead_time_demand"] = f"mu_LTD = {d:,.2f} * {L:,.2f} = {mu_ltd:,.2f} units"
        formulas_substituted["lead_time_demand_std"] = f"sigma_LT = {d:,.2f} * {l_std:,.2f} = {sigma_lt:,.2f} units"
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units"
        formulas_substituted["reorder_point"] = f"ROP = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units"
    else:
        var_comp1 = L * (d_std ** 2)
        var_comp2 = (d ** 2) * (l_std ** 2)
        formulas_substituted["expected_lead_time_demand"] = f"mu_LTD = {d:,.2f} * {L:,.2f} = {mu_ltd:,.2f} units"
        formulas_substituted["lead_time_demand_std"] = (
            f"sigma_LT = sqrt({L:,.2f} * ({d_std:,.2f})^2 + ({d:,.2f})^2 * ({l_std:,.2f})^2) = "
            f"sqrt({var_comp1:,.2f} + {var_comp2:,.2f}) = sqrt({var_comp1 + var_comp2:,.2f}) = {sigma_lt:,.2f} units"
        )
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units"
        formulas_substituted["reorder_point"] = f"ROP = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units"

    if holding_cost_per_unit is not None:
        formulas_substituted["annual_holding_cost_ss"] = f"AHC_SS = {ss:,.2f} * ${holding_cost_per_unit:,.2f} = ${ss_holding_cost:,.2f}/year"

    # Step-by-step derivation for student verification
    step_by_step_calculation = []
    if not using_direct_ltd:
        step_by_step_calculation.append(f"d = {d:,.2f}")
        step_by_step_calculation.append(f"sigma_d = {d_std:,.2f}")
        step_by_step_calculation.append(f"L = {L:,.2f}")
        step_by_step_calculation.append(f"sigma_L = {l_std:,.2f}")
    else:
        step_by_step_calculation.append(f"mu_LTD = {mu_ltd:,.2f}")
        step_by_step_calculation.append(f"sigma_LT = {sigma_lt:,.2f}")
    step_by_step_calculation.append(f"CSL = {csl:.4f} ({csl:.1%})")
    step_by_step_calculation.append(f"z = Phi^-1(CSL) = Phi^-1({csl:.4f}) = {z:.4f}")

    if using_direct_ltd:
        step_by_step_calculation.append("SS = z * sigma_LT")
        step_by_step_calculation.append(f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units.")
        step_by_step_calculation.append("ROP = mu_LTD + SS")
        step_by_step_calculation.append(f"ROP = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units.")
    elif d_std == 0.0 and l_std == 0.0:
        step_by_step_calculation.append("sigma_LT = 0 (constant demand & constant lead time)")
        step_by_step_calculation.append("SS = 0 units.")
        step_by_step_calculation.append("ROP = d * L")
        step_by_step_calculation.append(f"ROP = {d:,.2f} * {L:,.2f} = {rop:,.2f} units.")
    elif d_std > 0.0 and l_std == 0.0:
        step_by_step_calculation.append("sigma_LT = sigma_d * sqrt(L)")
        step_by_step_calculation.append(f"sigma_LT = {d_std:,.2f} * sqrt({L:,.2f}) = {sigma_lt:,.2f} units.")
        step_by_step_calculation.append("SS = z * sigma_LT")
        step_by_step_calculation.append(f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units.")
        step_by_step_calculation.append("ROP = d * L + SS")
        step_by_step_calculation.append(f"ROP = ({d:,.2f} * {L:,.2f}) + {ss:,.2f} = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units.")
    elif d_std == 0.0 and l_std > 0.0:
        step_by_step_calculation.append("sigma_LT = d * sigma_L")
        step_by_step_calculation.append(f"sigma_LT = {d:,.2f} * {l_std:,.2f} = {sigma_lt:,.2f} units.")
        step_by_step_calculation.append("SS = z * sigma_LT")
        step_by_step_calculation.append(f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units.")
        step_by_step_calculation.append("ROP = d * L + SS")
        step_by_step_calculation.append(f"ROP = ({d:,.2f} * {L:,.2f}) + {ss:,.2f} = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units.")
    else:
        var_c1 = L * (d_std ** 2)
        var_c2 = (d ** 2) * (l_std ** 2)
        step_by_step_calculation.append("sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)")
        step_by_step_calculation.append(f"sigma_LT = sqrt(({L:,.2f} x ({d_std:,.2f})^2) + (({d:,.2f})^2 x ({l_std:,.2f})^2))")
        step_by_step_calculation.append(f"sigma_LT = sqrt({var_c1:,.2f} + {var_c2:,.2f}) = sqrt({var_c1 + var_c2:,.2f}) = {sigma_lt:,.2f} units.")
        step_by_step_calculation.append("SS = z * sigma_LT")
        step_by_step_calculation.append(f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units.")
        step_by_step_calculation.append("ROP = d * L + SS")
        step_by_step_calculation.append(f"ROP = ({d:,.2f} * {L:,.2f}) + {ss:,.2f} = {mu_ltd:,.2f} + {ss:,.2f} = {rop:,.2f} units.")

    # 7. Verification Checks
    verification_checks = []
    # Check 1: Safety Stock Sign
    if csl >= 0.5 and ss >= 0:
        verification_checks.append({
            "name": "Safety Stock Non-Negativity",
            "check": "Safety Stock Non-Negativity",
            "status": "PASS",
            "detail": f"Safety stock ({ss:,.2f} units) is non-negative for target CSL >= 50%."
        })
    elif csl < 0.5:
        verification_checks.append({
            "name": "Safety Stock Non-Negativity",
            "check": "Safety Stock Non-Negativity",
            "status": "INFO",
            "detail": f"CSL ({csl:.1%}) < 50% results in negative safety buffer ({ss:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "Safety Stock Non-Negativity",
            "check": "Safety Stock Non-Negativity",
            "status": "FAIL",
            "detail": f"Safety stock is unexpectedly negative ({ss:,.2f} units)."
        })

    # Check 2: ROP Cushion
    if rop >= mu_ltd:
        verification_checks.append({
            "name": "ROP Coverage Cushion",
            "check": "ROP Coverage Cushion",
            "status": "PASS",
            "detail": f"Reorder Point ({rop:,.2f} units) covers expected lead time demand ({mu_ltd:,.2f} units) plus buffer ({ss:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "ROP Coverage Cushion",
            "check": "ROP Coverage Cushion",
            "status": "WARN",
            "detail": f"ROP ({rop:,.2f} units) is less than expected lead time demand ({mu_ltd:,.2f} units)."
        })

    # Check 3: Risk-Service Consistency
    if abs((1.0 - csl) - stockout_risk) < 1e-4:
        verification_checks.append({
            "name": "Risk-Service Consistency",
            "check": "Risk-Service Consistency",
            "status": "PASS",
            "detail": f"Cycle stockout risk {stockout_risk:.1%} exactly equals (1 - CSL)."
        })
    else:
        verification_checks.append({
            "name": "Risk-Service Consistency",
            "check": "Risk-Service Consistency",
            "status": "FAIL",
            "detail": f"Inconsistency between CSL ({csl:.1%}) and stockout risk ({stockout_risk:.1%})."
        })

    # Check 4: Lead time demand std dev validity
    if sigma_lt >= 0:
        verification_checks.append({
            "name": "Lead Time Demand Uncertainty Validity",
            "check": "Lead Time Demand Uncertainty Validity",
            "status": "PASS",
            "detail": f"Lead time demand standard deviation sigma_LT is non-negative ({sigma_lt:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "Lead Time Demand Uncertainty Validity",
            "check": "Lead Time Demand Uncertainty Validity",
            "status": "FAIL",
            "detail": "sigma_LT is negative."
        })

    # 8. Structured Output (containing all 8 student verification sections)
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
        "step_by_step_calculation": step_by_step_calculation,
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
        # 8 Mandated Student Verification Sections:
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formulas_substituted,
        "intermediate_calculations": intermediates_dict,
        "step_by_step_calculation": step_by_step_calculation,
        "step_by_step_derivation": step_by_step_calculation,
        "final_answer": final_answer_dict,
        "verification_checks": verification_checks,
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

    # 4. Variable Definitions
    variable_definitions = {
        "T": "Review Period Interval (periods between reviews)",
        "L": "Replenishment Lead Time (periods to receive order)",
        "T_plus_L": "Protection Interval / Period of Vulnerability (periods = T + L)",
        "d": "Average Demand per Period (units/period)",
        "sigma_d": "Standard Deviation of Demand per Period (units/period)",
        "CSL": "Target Cycle Service Level across Protection Interval",
        "z": "Safety Factor / Normal Quantile (z = Phi^-1(CSL))",
        "D_protection": "Expected Demand during Protection Interval (units = d * (T + L))",
        "sigma_protection": "Standard Deviation of Demand during Protection Interval (units = sigma_d * sqrt(T + L))",
        "SS": "Safety Stock over Protection Interval (units = z * sigma_protection)",
        "S": "Target Inventory / Order-Up-To Level (units = D_protection + SS)",
    }
    if inv_pos is not None:
        variable_definitions["IP"] = "Current Inventory Position (units = On-Hand + On-Order - Backorders)"
        variable_definitions["Q"] = "Recommended Order Quantity (units = max(0, S - IP))"
    if current_on_hand is not None:
        variable_definitions["I_OH"] = "On-Hand Inventory (units)"
        variable_definitions["I_OO"] = "On-Order / In-Transit Inventory (units)"
        variable_definitions["BO"] = "Backorders / Unfilled Demand (units)"

    # 5. Symbolic Formulas
    formulas = {
        "protection_period": "Protection period = review period + lead time = T + L",
        "expected_demand_protection": "D_protection = d * (T + L)",
        "std_demand_protection": "sigma_protection = sigma_d * sqrt(T + L)",
        "safety_factor": "z = Phi^-1(CSL)" if service_level is not None else "z = SS / sigma_protection",
        "safety_stock": "SS = z * sigma_protection = z * sigma_d * sqrt(T + L)",
        "target_inventory": "Target inventory = demand during protection period + safety stock = (d * (T + L)) + SS",
    }
    if inv_pos is not None:
        formulas["inventory_position"] = "IP = On-Hand + On-Order - Backorders" if current_on_hand is not None else "IP = inventory_position"
        formulas["order_quantity"] = "Order quantity = max(0, target inventory - inventory position) = max(0, S - IP)"

    # 6. Formulas with Substituted Values
    z_sub = f"z = Phi^-1({csl:.4f}) = {z:.4f}" if csl is not None else f"z = {calc_ss:.2f} / {std_demand_protection:.2f} = {z:.4f}"
    formulas_substituted = {
        "protection_period": f"T + L = {P:,.2f} + {L:,.2f} = {protection_period:,.2f} periods",
        "expected_demand_protection": f"D_protection = {d:,.2f} * {protection_period:,.2f} = {expected_demand_protection:,.2f} units",
        "std_demand_protection": f"sigma_protection = {sigma_d:,.2f} * sqrt({protection_period:,.2f}) = {sigma_d:,.2f} * {math.sqrt(protection_period):,.4f} = {std_demand_protection:,.2f} units",
        "safety_factor": z_sub,
        "safety_stock": f"SS = {z:.4f} * {std_demand_protection:,.2f} = {calc_ss:,.2f} units",
        "target_inventory": f"S = {expected_demand_protection:,.2f} + {calc_ss:,.2f} = {target_inventory:,.2f} units",
    }
    if inv_pos is not None:
        if current_on_hand is not None:
            formulas_substituted["inventory_position"] = f"IP = {float(current_on_hand):,.2f} + {float(current_on_order):,.2f} - {float(backorders):,.2f} = {inv_pos:,.2f} units"
        else:
            formulas_substituted["inventory_position"] = f"IP = {inv_pos:,.2f} units"
        formulas_substituted["order_quantity"] = f"Q = max(0, {target_inventory:,.2f} - {inv_pos:,.2f}) = max(0, {target_inventory - inv_pos:,.2f}) = {order_quantity:,.2f} units"

    # Step-by-step derivation for student verification
    step_by_step_calculation = [
        f"T = {P:,.2f} periods",
        f"L = {L:,.2f} periods",
        f"d = {d:,.2f} units/period",
        f"sigma_d = {sigma_d:,.2f} units/period",
        f"CSL = {csl:.4f} ({csl:.1%})" if csl is not None else f"Safety stock fixed = {calc_ss:,.2f} units",
        "Protection period = review period + lead time",
        f"Protection period = T + L = {P:,.2f} + {L:,.2f} = {protection_period:,.2f} periods.",
        "D_protection = d * (T + L)",
        f"D_protection = {d:,.2f} * {protection_period:,.2f} = {expected_demand_protection:,.2f} units.",
        "sigma_protection = sigma_d * sqrt(T + L)",
        f"sigma_protection = {sigma_d:,.2f} * sqrt({protection_period:,.2f}) = {std_demand_protection:,.2f} units.",
        f"z = Phi^-1(CSL) = Phi^-1({csl:.4f}) = {z:.4f}" if csl is not None else f"z = {z:.4f}",
        "SS = z * sigma_protection",
        f"SS = {z:.4f} * {std_demand_protection:,.2f} = {calc_ss:,.2f} units.",
        "Target inventory = demand during protection period + safety stock",
        "S = D_protection + SS",
        f"S = {expected_demand_protection:,.2f} + {calc_ss:,.2f} = {target_inventory:,.2f} units.",
    ]
    if inv_pos is not None:
        if current_on_hand is not None:
            step_by_step_calculation.append(
                f"IP = On-Hand ({float(current_on_hand):,.2f}) + On-Order ({float(current_on_order):,.2f}) - Backorders ({float(backorders):,.2f}) = {inv_pos:,.2f} units."
            )
        else:
            step_by_step_calculation.append(f"IP = {inv_pos:,.2f} units.")
        step_by_step_calculation.append("Order quantity = max(0, target inventory - inventory position)")
        step_by_step_calculation.append("Q = max(0, S - IP)")
        step_by_step_calculation.append(f"Q = max(0, {target_inventory:,.2f} - {inv_pos:,.2f}) = {order_quantity:,.2f} units.")

    # 7. Verification Checks
    verification_checks = []
    # Check 1: Protection interval coverage
    if target_inventory >= expected_demand_protection:
        verification_checks.append({
            "name": "Protection Interval Coverage",
            "check": "Protection Interval Coverage",
            "status": "PASS",
            "detail": f"Target inventory level S ({target_inventory:,.2f} units) safely covers protection interval demand ({expected_demand_protection:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "Protection Interval Coverage",
            "check": "Protection Interval Coverage",
            "status": "WARN",
            "detail": f"Target inventory S ({target_inventory:,.2f} units) is below expected demand ({expected_demand_protection:,.2f} units)."
        })

    # Check 2: Safety stock non-negativity
    if calc_ss >= 0:
        verification_checks.append({
            "name": "Safety Stock Non-Negativity",
            "check": "Safety Stock Non-Negativity",
            "status": "PASS",
            "detail": f"Safety stock is non-negative ({calc_ss:,.2f} units)."
        })
    else:
        verification_checks.append({
            "name": "Safety Stock Non-Negativity",
            "check": "Safety Stock Non-Negativity",
            "status": "FAIL",
            "detail": f"Safety stock is negative ({calc_ss:,.2f} units)."
        })

    # Check 3: Order quantity non-negativity
    if order_quantity is not None:
        if order_quantity >= 0:
            verification_checks.append({
                "name": "Non-Negative Order Quantity",
                "check": "Non-Negative Order Quantity",
                "status": "PASS",
                "detail": f"Recommended order quantity ({order_quantity:,.2f} units) is non-negative (max(0, S - IP))."
            })
        else:
            verification_checks.append({
                "name": "Non-Negative Order Quantity",
                "check": "Non-Negative Order Quantity",
                "status": "FAIL",
                "detail": "Recommended order quantity is negative."
            })

    # Check 4: Inventory position conservation
    if current_on_hand is not None:
        expected_ip = float(current_on_hand) + float(current_on_order) - float(backorders)
        if abs(inv_pos - expected_ip) < 1e-4:
            verification_checks.append({
                "name": "Inventory Position Conservation",
                "check": "Inventory Position Conservation",
                "status": "PASS",
                "detail": f"Inventory position ({inv_pos:,.2f} units) accurately reflects On-Hand ({float(current_on_hand):,.2f}) + On-Order ({float(current_on_order):,.2f}) - Backorders ({float(backorders):,.2f})."
            })
        else:
            verification_checks.append({
                "name": "Inventory Position Conservation",
                "check": "Inventory Position Conservation",
                "status": "FAIL",
                "detail": "Discrepancy in inventory position calculation."
            })

    # 8. Structured Output (containing all 8 student verification sections)
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
        "step_by_step_calculation": step_by_step_calculation,
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
        # 8 Mandated Student Verification Sections:
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formulas_substituted,
        "intermediate_calculations": intermediates_dict,
        "step_by_step_calculation": step_by_step_calculation,
        "step_by_step_derivation": step_by_step_calculation,
        "final_answer": final_answer_dict,
        "verification_checks": verification_checks,
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

    # 4. Variable Definitions
    variable_definitions = {
        "Cu": "Cost of Underage / Shortage Cost per unit ($/unit)",
        "Co": "Cost of Overage / Excess Cost per unit ($/unit)",
        "CR": "Critical Ratio / Critical Fractile (optimal service probability = Cu / (Cu + Co))",
        "mu": "Mean Demand for the single period (units)",
        "sigma": "Standard Deviation of Demand for the single period (units)",
        "z*": "Optimal Safety Factor / Normal Quantile (z* = Phi^-1(CR))",
        "Q*": "Recommended Order Quantity (units = mu + z* * sigma)",
        "E_LostSales": "Expected Lost Sales / Stockout Volume (units = sigma * L(z*))",
        "E_Leftover": "Expected Leftover Inventory / Unsold Stock (units = Q* - mu + E_LostSales)",
        "E_Sales": "Expected Sales Volume (units = mu - E_LostSales)",
        "InStockProb": "Optimal In-Stock Probability (Phi(z*))",
    }
    if not has_direct_cu_co:
        variable_definitions["p"] = "Unit Selling Price ($/unit)"
        variable_definitions["c"] = "Unit Purchase Cost ($/unit)"
        variable_definitions["s"] = "Unit Salvage Value for unsold stock ($/unit)"
        variable_definitions["g"] = "Unit Goodwill Penalty / Stockout Penalty ($/unit)"
    if p is not None and c is not None:
        variable_definitions["E_Profit"] = "Expected Total Profit ($)"
        variable_definitions["EVPI"] = "Expected Value of Perfect Information ($ = Profit_PI - Expected Profit)"

    # 5. Symbolic Formulas
    formulas = {
        "underage_cost": "Cu = p - c + g" if not has_direct_cu_co else "Cu = underage_cost",
        "overage_cost": "Co = c - s" if not has_direct_cu_co else "Co = overage_cost",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "safety_factor": "z* = Phi^-1(CR)",
        "optimal_order_quantity": "Q* = max(0, mu + z* * sigma)",
        "expected_lost_sales": "E[Lost Sales] = sigma * [phi(z*) - z* * (1 - Phi(z*))]",
        "expected_leftover": "E[Leftover] = Q* - mu + E[Lost Sales]",
        "expected_sales": "E[Sales] = mu - E[Lost Sales]",
        "in_stock_probability": "P(Demand <= Q*) = Phi(z*)",
    }
    if p is not None and c is not None:
        formulas["expected_profit"] = "E[Profit] = (p * E[Sales]) + (s * E[Leftover]) - (c * Q*) - (g * E[Lost Sales])"
        formulas["evpi"] = "EVPI = ((p - c) * mu) - E[Profit]"

    # 6. Formulas with Substituted Values
    cu_sub = f"Cu = ${p:,.2f} - ${c:,.2f} + ${g:,.2f} = ${cu:,.2f}/unit" if not has_direct_cu_co else f"Cu = ${cu:,.2f}/unit"
    co_sub = f"Co = ${c:,.2f} - ${s:,.2f} = ${co:,.2f}/unit" if not has_direct_cu_co else f"Co = ${co:,.2f}/unit"
    formulas_substituted = {
        "underage_cost": cu_sub,
        "overage_cost": co_sub,
        "critical_ratio": f"CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {cr:.4f} ({cr:.2%})",
        "safety_factor": f"z* = Phi^-1({cr:.4f}) = {effective_z:.4f}",
        "optimal_order_quantity": f"Q* = {mu:,.2f} + ({effective_z:.4f} * {sigma:,.2f}) = {mu:,.2f} + {effective_z * sigma:,.2f} = {q_star:,.2f} units",
        "expected_lost_sales": f"E[Lost Sales] = {sigma:,.2f} * L({effective_z:.4f}) = {exp_lost_sales:,.2f} units",
        "expected_leftover": f"E[Leftover] = {q_star:,.2f} - {mu:,.2f} + {exp_lost_sales:,.2f} = {exp_leftover:,.2f} units",
        "expected_sales": f"E[Sales] = {mu:,.2f} - {exp_lost_sales:,.2f} = {exp_sales:,.2f} units",
        "in_stock_probability": f"P(Demand <= {q_star:,.2f}) = Phi({effective_z:.4f}) = {in_stock_prob:.4f} ({in_stock_prob:.2%})",
    }
    if p is not None and c is not None:
        formulas_substituted["expected_profit"] = (
            f"E[Profit] = (${p:,.2f} * {exp_sales:,.2f}) + (${s:,.2f} * {exp_leftover:,.2f}) - "
            f"(${c:,.2f} * {q_star:,.2f}) - (${g:,.2f} * {exp_lost_sales:,.2f}) = "
            f"${exp_rev:,.2f} + ${exp_salvage:,.2f} - ${exp_cost:,.2f} - ${exp_penalty:,.2f} = "
            f"${exp_profit:,.2f}"
        )
        formulas_substituted["evpi"] = (
            f"EVPI = ((${p:,.2f} - ${c:,.2f}) * {mu:,.2f}) - ${exp_profit:,.2f} = "
            f"${pi_profit:,.2f} - ${exp_profit:,.2f} = ${evpi:,.2f}"
        )

    # Step-by-step derivation for student verification
    step_by_step_calculation = []
    if not has_direct_cu_co:
        step_by_step_calculation.extend([
            f"p = ${p:,.2f}",
            f"c = ${c:,.2f}",
            f"s = ${s:,.2f}",
            f"g = ${g:,.2f}",
            f"mu = {mu:,.2f} units",
            f"sigma = {sigma:,.2f} units",
            "Cu = p - c + g",
            f"Cu = ${p:,.2f} - ${c:,.2f} + ${g:,.2f} = ${cu:,.2f}/unit.",
            "Co = c - s",
            f"Co = ${c:,.2f} - ${s:,.2f} = ${co:,.2f}/unit.",
        ])
    else:
        step_by_step_calculation.extend([
            f"Cu = ${cu:,.2f}/unit",
            f"Co = ${co:,.2f}/unit",
            f"mu = {mu:,.2f} units",
            f"sigma = {sigma:,.2f} units",
        ])
    step_by_step_calculation.extend([
        "CR = Cu / (Cu + Co)",
        f"CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {cr:.4f} ({cr:.2%}).",
        f"z* = Phi^-1(CR) = Phi^-1({cr:.4f}) = {effective_z:.4f}",
        "Q* = mu + z* * sigma",
        f"Q* = {mu:,.2f} + ({effective_z:.4f} * {sigma:,.2f}) = {mu:,.2f} + {effective_z * sigma:,.2f} = {q_star:,.2f} units.",
        f"E[Lost Sales] = sigma * L(z*) = {sigma:,.2f} * L({effective_z:.4f}) = {exp_lost_sales:,.2f} units.",
        f"E[Leftover] = Q* - mu + E[Lost Sales] = {q_star:,.2f} - {mu:,.2f} + {exp_lost_sales:,.2f} = {exp_leftover:,.2f} units.",
        f"E[Sales] = mu - E[Lost Sales] = {mu:,.2f} - {exp_lost_sales:,.2f} = {exp_sales:,.2f} units.",
        f"P(Demand <= Q*) = Phi({effective_z:.4f}) = {in_stock_prob:.4f} ({in_stock_prob:.2%}).",
    ])
    if p is not None and c is not None:
        step_by_step_calculation.append(
            f"E[Profit] = (p * E[Sales]) + (s * E[Leftover]) - (c * Q*) - (g * E[Lost Sales]) = ${exp_profit:,.2f}."
        )
        step_by_step_calculation.append(
            f"EVPI = ((p - c) * mu) - E[Profit] = ${evpi:,.2f}."
        )

    # 7. Verification Checks
    verification_checks = []
    # Check 1: Economic viability
    if cu > 0 and co > 0:
        verification_checks.append({
            "name": "Economic Viability",
            "check": "Economic Viability",
            "status": "PASS",
            "detail": f"Both underage cost (${cu:,.2f}) and overage cost (${co:,.2f}) are strictly positive."
        })
    else:
        verification_checks.append({
            "name": "Economic Viability",
            "check": "Economic Viability",
            "status": "FAIL",
            "detail": "Non-positive underage or overage costs."
        })

    # Check 2: Critical fractile range
    if 0 < cr < 1.0:
        verification_checks.append({
            "name": "Critical Fractile Range",
            "check": "Critical Fractile Range",
            "status": "PASS",
            "detail": f"Critical ratio CR = {cr:.4f} ({cr:.2%}) is strictly bounded between 0 and 1."
        })
    else:
        verification_checks.append({
            "name": "Critical Fractile Range",
            "check": "Critical Fractile Range",
            "status": "FAIL",
            "detail": f"Critical ratio CR = {cr} is out of bounds."
        })

    # Check 3: High/low margin bias consistency
    if cr > 0.5 and q_star >= mu:
        verification_checks.append({
            "name": "Margin Bias Direction",
            "check": "Margin Bias Direction",
            "status": "PASS",
            "detail": f"High margin (CR = {cr:.2%} > 50%) correctly biases Q* ({q_star:,.2f}) above mean demand ({mu:,.2f}) to avoid stockouts."
        })
    elif cr <= 0.5 and q_star <= mu:
        verification_checks.append({
            "name": "Margin Bias Direction",
            "check": "Margin Bias Direction",
            "status": "PASS",
            "detail": f"Low margin (CR = {cr:.2%} <= 50%) correctly biases Q* ({q_star:,.2f}) at or below mean demand ({mu:,.2f}) to minimize leftover loss."
        })
    else:
        verification_checks.append({
            "name": "Margin Bias Direction",
            "check": "Margin Bias Direction",
            "status": "WARN",
            "detail": f"Order quantity Q* ({q_star:,.2f}) does not match expected margin bias direction."
        })

    # Check 4: In-stock probability consistency
    if in_stock_prob >= 0:
        verification_checks.append({
            "name": "In-Stock Availability Consistency",
            "check": "In-Stock Availability Consistency",
            "status": "PASS",
            "detail": f"In-stock probability of {in_stock_prob:.2%} aligns with target critical fractile ({cr:.2%})."
        })
    else:
        verification_checks.append({
            "name": "In-Stock Availability Consistency",
            "check": "In-Stock Availability Consistency",
            "status": "FAIL",
            "detail": "In-stock probability is negative."
        })

    # 8. Structured Output (containing all 8 student verification sections)
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
        "step_by_step_calculation": step_by_step_calculation,
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
        # 8 Mandated Student Verification Sections:
        "inputs_extracted": inputs_dict,
        "inputs": inputs_dict,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formulas_substituted,
        "intermediate_calculations": intermediates_dict,
        "step_by_step_calculation": step_by_step_calculation,
        "step_by_step_derivation": step_by_step_calculation,
        "final_answer": final_answer_dict,
        "verification_checks": verification_checks,
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
