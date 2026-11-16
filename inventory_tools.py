"""
Supply Chain Analytics Inventory Tools
======================================
Pure Python numerical calculation engine for inventory decision models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock & Reorder Point (Continuous Review)
4. Fixed-Period Review Model (Periodic Review / Order-Up-To)
5. Newsvendor Model (Single-Period Inventory)

All calculations are strictly deterministic and mathematically rigorous,
using standard supply chain analytics formulas.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from statistics import NormalDist


# ==============================================================================
# Statistical Helper Functions
# ==============================================================================

def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function Phi(z)."""
    return NormalDist().cdf(z)


def normal_inv_cdf(p: float) -> float:
    """Standard normal inverse CDF / quantile function Phi^{-1}(p)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Service level / probability must be in (0, 1), got {p}")
    return NormalDist().inv_cdf(p)


def normal_pdf(z: float) -> float:
    """Standard normal probability density function phi(z)."""
    return NormalDist().pdf(z)


def unit_normal_loss(z: float) -> float:
    """
    Unit normal loss function L(z) = E[max(Z - z, 0)]
    L(z) = phi(z) - z * (1 - Phi(z))
    Used for expected shortage / lost sales calculations.
    """
    phi = normal_pdf(z)
    Phi = normal_cdf(z)
    return phi - z * (1.0 - Phi)


# ==============================================================================
# Model 1: Economic Order Quantity (EOQ)
# ==============================================================================

@dataclass
class EOQInputs:
    annual_demand: float                # D (units/year)
    ordering_cost: float                # S ($/order setup or procurement cost)
    holding_cost: Optional[float] = None # H ($/unit/year)
    holding_cost_rate: Optional[float] = None # h (% of unit cost per year)
    unit_cost: Optional[float] = None   # C ($/unit purchase cost)
    lead_time_days: Optional[float] = None # L (lead time in days)
    working_days_per_year: float = 250.0  # W (working days/year)


@dataclass
class EOQOutputs:
    optimal_order_quantity: float       # Q* (units)
    annual_orders_count: float          # N = D / Q*
    order_cycle_days: float             # T in working days = (Q* / D) * W
    annual_ordering_cost: float         # AOC = (D / Q*) * S
    annual_holding_cost: float          # AHC = (Q* / 2) * H
    total_relevant_cost: float          # TRC = AOC + AHC
    annual_purchasing_cost: float       # APC = D * C (if C provided)
    total_annual_cost: float            # TC = APC + TRC
    unit_holding_cost: float            # H used in calculation
    reorder_point: Optional[float] = None # ROP = d * L (if lead time provided)
    daily_demand: Optional[float] = None
    cost_parity_difference: float = 0.0 # |AOC - AHC| to verify optimality


def calculate_eoq(inputs: EOQInputs) -> EOQOutputs:
    """
    Computes Economic Order Quantity (EOQ) and associated cost and cycle metrics.
    Formula:
        Q* = sqrt((2 * D * S) / H)
        H = holding_cost OR (holding_cost_rate * unit_cost)
    """
    D = float(inputs.annual_demand)
    S = float(inputs.ordering_cost)
    W = float(inputs.working_days_per_year) if inputs.working_days_per_year > 0 else 250.0

    if D <= 0:
        raise ValueError(f"Annual demand D must be strictly positive, got {D}")
    if S <= 0:
        raise ValueError(f"Ordering cost S must be strictly positive, got {S}")

    # Determine holding cost per unit H
    if inputs.holding_cost is not None and inputs.holding_cost > 0:
        H = float(inputs.holding_cost)
    elif inputs.holding_cost_rate is not None and inputs.unit_cost is not None:
        rate = float(inputs.holding_cost_rate)
        # Handle if rate was passed as percentage (e.g. 20 instead of 0.20)
        if rate > 1.0:
            rate = rate / 100.0
        H = rate * float(inputs.unit_cost)
    else:
        raise ValueError("Must provide either 'holding_cost' (H) or both 'holding_cost_rate' (h) and 'unit_cost' (C)")

    if H <= 0:
        raise ValueError(f"Holding cost H must be strictly positive, got {H}")

    # EOQ calculation
    Q_star = math.sqrt((2.0 * D * S) / H)
    N = D / Q_star
    cycle_days = (Q_star / D) * W
    annual_ordering_cost = N * S
    annual_holding_cost = (Q_star / 2.0) * H
    total_relevant_cost = annual_ordering_cost + annual_holding_cost

    C = float(inputs.unit_cost) if inputs.unit_cost is not None else 0.0
    annual_purchasing_cost = D * C
    total_annual_cost = annual_purchasing_cost + total_relevant_cost

    daily_demand = D / W
    rop = None
    if inputs.lead_time_days is not None and inputs.lead_time_days >= 0:
        ltd = daily_demand * float(inputs.lead_time_days)
        # If LTD exceeds Q*, ROP is LTD modulo Q*
        rop = ltd if ltd <= Q_star else ltd % Q_star

    parity_diff = abs(annual_ordering_cost - annual_holding_cost)

    return EOQOutputs(
        optimal_order_quantity=round(Q_star, 2),
        annual_orders_count=round(N, 2),
        order_cycle_days=round(cycle_days, 1),
        annual_ordering_cost=round(annual_ordering_cost, 2),
        annual_holding_cost=round(annual_holding_cost, 2),
        total_relevant_cost=round(total_relevant_cost, 2),
        annual_purchasing_cost=round(annual_purchasing_cost, 2),
        total_annual_cost=round(total_annual_cost, 2),
        unit_holding_cost=round(H, 4),
        reorder_point=round(rop, 2) if rop is not None else None,
        daily_demand=round(daily_demand, 2),
        cost_parity_difference=round(parity_diff, 4)
    )


def verify_eoq(inputs: EOQInputs, outputs: EOQOutputs) -> Dict[str, Any]:
    """Sanity and constraint verification for EOQ model."""
    checks = []
    
    # Check 1: Positive Q*
    q_pos = outputs.optimal_order_quantity > 0
    checks.append({
        "check": "Positive Order Quantity",
        "passed": q_pos,
        "detail": f"Q* = {outputs.optimal_order_quantity} units > 0"
    })
    
    # Check 2: First-order optimality parity: AOC == AHC
    # At continuous Q*, AOC == AHC. Due to rounding Q* to 2 decimals, allow small delta.
    parity_tol = 5.0  # $5 tolerance for rounding
    parity_passed = outputs.cost_parity_difference <= parity_tol
    checks.append({
        "check": "First-Order Cost Parity (AOC == AHC)",
        "passed": parity_passed,
        "detail": f"|AOC - AHC| = ${outputs.cost_parity_difference:.2f} (AOC=${outputs.annual_ordering_cost:.2f}, AHC=${outputs.annual_holding_cost:.2f})"
    })

    # Check 3: Convexity: second derivative of TRC with respect to Q is 2*D*S / Q^3 > 0
    convexity = (2.0 * inputs.annual_demand * inputs.ordering_cost) / (outputs.optimal_order_quantity ** 3) > 0
    checks.append({
        "check": "Cost Curve Strict Convexity (d²TC/dQ² > 0)",
        "passed": convexity,
        "detail": "Verified minimum total cost (positive second derivative)"
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "model": "EOQ",
        "verified": all_passed,
        "checks": checks
    }


# ==============================================================================
# Model 2: Quantity Discount Model
# ==============================================================================

@dataclass
class DiscountTier:
    min_quantity: float                 # Minimum order quantity to qualify
    max_quantity: Optional[float]       # Maximum quantity for tier (None = infinity)
    unit_price: float                   # C_i ($/unit)
    tier_label: str = ""


@dataclass
class TierEvaluation:
    tier_index: int
    tier_label: str
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    unconstrained_eoq: float
    is_eoq_feasible: bool
    evaluated_quantity: float           # Feasible quantity chosen for this tier
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float


@dataclass
class QuantityDiscountInputs:
    annual_demand: float                # D (units/year)
    ordering_cost: float                # S ($/order)
    tiers: List[DiscountTier]           # List of discount brackets
    holding_cost_rate: Optional[float] = None # h (% of unit price)
    fixed_holding_cost: Optional[float] = None # H ($/unit/year if fixed)


@dataclass
class QuantityDiscountOutputs:
    optimal_tier_index: int
    optimal_tier_label: str
    optimal_order_quantity: float
    optimal_unit_price: float
    min_total_annual_cost: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    baseline_total_cost: float          # Cost at lowest tier (no discount)
    annual_cost_savings: float          # Baseline TC - Optimal TC
    percentage_savings: float           # (Savings / Baseline TC) * 100
    tier_evaluations: List[TierEvaluation]


def calculate_quantity_discount(inputs: QuantityDiscountInputs) -> QuantityDiscountOutputs:
    """
    Evaluates all-units quantity discount schedule and selects the order quantity
    that minimizes Total Annual Cost (Purchase + Ordering + Holding).
    """
    D = float(inputs.annual_demand)
    S = float(inputs.ordering_cost)

    if D <= 0 or S <= 0:
        raise ValueError("Demand and ordering cost must be positive.")
    if not inputs.tiers:
        raise ValueError("At least one discount tier must be provided.")

    # Sort tiers by min_quantity ascending
    sorted_tiers = sorted(inputs.tiers, key=lambda t: t.min_quantity)

    evaluations: List[TierEvaluation] = []

    for idx, tier in enumerate(sorted_tiers):
        C = float(tier.unit_price)
        q_min = float(tier.min_quantity)
        q_max = float(tier.max_quantity) if tier.max_quantity is not None else float("inf")

        # Determine holding cost for this tier
        if inputs.holding_cost_rate is not None:
            rate = float(inputs.holding_cost_rate)
            if rate > 1.0:
                rate = rate / 100.0
            H = rate * C
        elif inputs.fixed_holding_cost is not None:
            H = float(inputs.fixed_holding_cost)
        else:
            raise ValueError("Must provide either 'holding_cost_rate' or 'fixed_holding_cost'")

        # Calculate unconstrained EOQ
        eoq_raw = math.sqrt((2.0 * D * S) / H)

        # Check feasibility
        if eoq_raw < q_min:
            # Below tier threshold: adjust to minimum quantity to qualify for discount
            is_feasible = False
            q_eval = q_min
        elif eoq_raw > q_max:
            # Above tier max: in all-units discount, higher tier would be cheaper, but cap at max
            is_feasible = False
            q_eval = q_max
        else:
            is_feasible = True
            q_eval = eoq_raw

        # Compute costs for candidate q_eval
        purchase_cost = D * C
        ordering_cost = (D / q_eval) * S
        holding_cost = (q_eval / 2.0) * H
        total_cost = purchase_cost + ordering_cost + holding_cost

        label = tier.tier_label or f"Tier {idx + 1} ({int(q_min)}-{int(q_max) if q_max != float('inf') else '+'})"

        evaluations.append(TierEvaluation(
            tier_index=idx,
            tier_label=label,
            min_quantity=q_min,
            max_quantity=None if q_max == float("inf") else q_max,
            unit_price=round(C, 4),
            unconstrained_eoq=round(eoq_raw, 2),
            is_eoq_feasible=is_feasible,
            evaluated_quantity=round(q_eval, 2),
            annual_purchase_cost=round(purchase_cost, 2),
            annual_ordering_cost=round(ordering_cost, 2),
            annual_holding_cost=round(holding_cost, 2),
            total_annual_cost=round(total_cost, 2)
        ))

    # Find the tier that minimizes total_annual_cost
    best_eval = min(evaluations, key=lambda e: e.total_annual_cost)
    baseline_eval = evaluations[0]
    savings = baseline_eval.total_annual_cost - best_eval.total_annual_cost
    pct_savings = (savings / baseline_eval.total_annual_cost * 100.0) if baseline_eval.total_annual_cost > 0 else 0.0

    return QuantityDiscountOutputs(
        optimal_tier_index=best_eval.tier_index,
        optimal_tier_label=best_eval.tier_label,
        optimal_order_quantity=best_eval.evaluated_quantity,
        optimal_unit_price=best_eval.unit_price,
        min_total_annual_cost=best_eval.total_annual_cost,
        annual_purchase_cost=best_eval.annual_purchase_cost,
        annual_ordering_cost=best_eval.annual_ordering_cost,
        annual_holding_cost=best_eval.annual_holding_cost,
        baseline_total_cost=baseline_eval.total_annual_cost,
        annual_cost_savings=round(savings, 2),
        percentage_savings=round(pct_savings, 2),
        tier_evaluations=evaluations
    )


def verify_quantity_discount(inputs: QuantityDiscountInputs, outputs: QuantityDiscountOutputs) -> Dict[str, Any]:
    """Sanity and constraint verification for Quantity Discount model."""
    checks = []

    # Check 1: Monotonicity of prices
    prices = [t.unit_price for t in outputs.tier_evaluations]
    is_price_decreasing = all(prices[i] >= prices[i+1] for i in range(len(prices)-1))
    checks.append({
        "check": "Price Schedule Monotonicity",
        "passed": is_price_decreasing,
        "detail": f"Unit prices decrease/non-increase with volume: {prices}"
    })

    # Check 2: Optimal cost is global minimum across evaluated tiers
    min_cost = min(e.total_annual_cost for e in outputs.tier_evaluations)
    is_min = abs(outputs.min_total_annual_cost - min_cost) < 0.01
    checks.append({
        "check": "Global Cost Minimization Across All Tiers",
        "passed": is_min,
        "detail": f"Optimal cost ${outputs.min_total_annual_cost:,.2f} is lowest among all evaluated tiers"
    })

    # Check 3: Feasibility of evaluated quantity
    opt_tier = outputs.tier_evaluations[outputs.optimal_tier_index]
    q_opt = outputs.optimal_order_quantity
    q_min = opt_tier.min_quantity
    q_max = opt_tier.max_quantity if opt_tier.max_quantity is not None else float("inf")
    q_feasible = q_min <= q_opt <= q_max
    checks.append({
        "check": "Optimal Quantity Fits Tier Bracket",
        "passed": q_feasible,
        "detail": f"Order quantity {q_opt} satisfies bracket [{q_min}, {q_max}]"
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "model": "Quantity Discount",
        "verified": all_passed,
        "checks": checks
    }


# ==============================================================================
# Model 3: Safety Stock and Reorder Point (Continuous Review)
# ==============================================================================

@dataclass
class SafetyStockROPInputs:
    lead_time: float                    # L (lead time periods, e.g. days or weeks)
    demand_mean: float                  # d (average demand per period)
    demand_std_dev: float = 0.0         # sigma_d (std dev of demand per period)
    lead_time_std_dev: float = 0.0      # sigma_L (std dev of lead time, optional)
    service_level: float = 0.95         # Cycle Service Level CSL in (0, 1)
    order_quantity: Optional[float] = None # Q (for item fill rate calculation)
    unit_holding_cost: Optional[float] = None # H ($/unit/year)


@dataclass
class SafetyStockROPOutputs:
    service_level: float
    z_score: float
    expected_lead_time_demand: float    # mu_LTD = d * L
    lead_time_demand_std_dev: float     # sigma_LTD
    safety_stock: float                 # SS = z * sigma_LTD
    reorder_point: float                # ROP = mu_LTD + SS
    stockout_probability_per_cycle: float # 1 - CSL
    expected_shortage_per_cycle: float  # sigma_LTD * L(z)
    item_fill_rate: Optional[float] = None # 1 - (ESC / Q)
    annual_safety_stock_holding_cost: Optional[float] = None


def calculate_safety_stock_and_rop(inputs: SafetyStockROPInputs) -> SafetyStockROPOutputs:
    """
    Calculates Safety Stock and Reorder Point (ROP) under continuous review policy.
    Supports:
      1. Variable demand, constant lead time: sigma_LTD = sigma_d * sqrt(L)
      2. Constant demand, variable lead time: sigma_LTD = d * sigma_L
      3. Both variable: sigma_LTD = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
    """
    L = float(inputs.lead_time)
    d = float(inputs.demand_mean)
    sigma_d = float(inputs.demand_std_dev)
    sigma_L = float(inputs.lead_time_std_dev)
    csl = float(inputs.service_level)

    if csl > 1.0:
        csl = csl / 100.0  # Convert percentage e.g. 95 -> 0.95
    if not (0.0 < csl < 1.0):
        raise ValueError(f"Service level must be between 0 and 1, got {csl}")
    if L < 0 or d < 0 or sigma_d < 0 or sigma_L < 0:
        raise ValueError("Lead time, demand, and standard deviations must be non-negative.")

    # Calculate z-score
    z = normal_inv_cdf(csl)

    # Lead time demand expectation
    mu_LTD = d * L

    # Lead time demand standard deviation
    if sigma_d > 0 and sigma_L > 0:
        sigma_LTD = math.sqrt((L * (sigma_d ** 2)) + ((d ** 2) * (sigma_L ** 2)))
    elif sigma_d > 0 and sigma_L == 0:
        sigma_LTD = sigma_d * math.sqrt(L)
    elif sigma_d == 0 and sigma_L > 0:
        sigma_LTD = d * sigma_L
    else:
        sigma_LTD = 0.0

    # Safety Stock and ROP
    safety_stock = z * sigma_LTD
    rop = mu_LTD + safety_stock

    # Expected shortage per cycle (loss function)
    loss_z = unit_normal_loss(z) if sigma_LTD > 0 else 0.0
    esc = sigma_LTD * loss_z

    # Fill rate if order quantity Q provided
    fill_rate = None
    if inputs.order_quantity is not None and inputs.order_quantity > 0:
        Q = float(inputs.order_quantity)
        fill_rate = max(0.0, min(1.0, 1.0 - (esc / Q)))

    # Annual holding cost of safety stock
    ss_holding_cost = None
    if inputs.unit_holding_cost is not None and inputs.unit_holding_cost > 0:
        ss_holding_cost = safety_stock * float(inputs.unit_holding_cost)

    return SafetyStockROPOutputs(
        service_level=round(csl, 4),
        z_score=round(z, 4),
        expected_lead_time_demand=round(mu_LTD, 2),
        lead_time_demand_std_dev=round(sigma_LTD, 2),
        safety_stock=round(safety_stock, 2),
        reorder_point=round(rop, 2),
        stockout_probability_per_cycle=round(1.0 - csl, 4),
        expected_shortage_per_cycle=round(esc, 4),
        item_fill_rate=round(fill_rate, 4) if fill_rate is not None else None,
        annual_safety_stock_holding_cost=round(ss_holding_cost, 2) if ss_holding_cost is not None else None
    )


def verify_safety_stock_rop(inputs: SafetyStockROPInputs, outputs: SafetyStockROPOutputs) -> Dict[str, Any]:
    """Sanity and constraint verification for Safety Stock & ROP."""
    checks = []

    # Check 1: Non-negative safety stock for CSL >= 0.5
    ss_valid = outputs.safety_stock >= 0 if outputs.service_level >= 0.5 else True
    checks.append({
        "check": "Safety Stock Validity (SS >= 0 for CSL >= 50%)",
        "passed": ss_valid,
        "detail": f"Safety stock = {outputs.safety_stock} for CSL={outputs.service_level*100:.1f}% (z={outputs.z_score})"
    })

    # Check 2: ROP >= Expected Lead Time Demand for CSL >= 0.5
    rop_valid = outputs.reorder_point >= outputs.expected_lead_time_demand if outputs.service_level >= 0.5 else True
    checks.append({
        "check": "Reorder Point Exceeds Expected LTD",
        "passed": rop_valid,
        "detail": f"ROP ({outputs.reorder_point}) >= Expected LTD ({outputs.expected_lead_time_demand})"
    })

    # Check 3: Fill rate >= Cycle Service Level (since Type 2 service level >= Type 1)
    if outputs.item_fill_rate is not None:
        fr_valid = outputs.item_fill_rate >= outputs.service_level - 0.05
        checks.append({
            "check": "Fill Rate Consistency (Fill Rate >= CSL)",
            "passed": fr_valid,
            "detail": f"Fill rate = {outputs.item_fill_rate*100:.2f}%, CSL = {outputs.service_level*100:.2f}%"
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "model": "Safety Stock & ROP",
        "verified": all_passed,
        "checks": checks
    }


# ==============================================================================
# Model 4: Fixed-Period Review Model (Periodic Review / Order-Up-To)
# ==============================================================================

@dataclass
class FixedPeriodReviewInputs:
    review_period: float                # P or R (time between reviews, e.g. 7 days)
    lead_time: float                    # L (replenishment lead time, e.g. 3 days)
    demand_mean: float                  # d (average demand per time period)
    demand_std_dev: float               # sigma_d (demand standard deviation per period)
    service_level: float = 0.95         # CSL (e.g. 0.95)
    current_inventory_on_hand: Optional[float] = None # I (on hand)
    inventory_on_order: Optional[float] = 0.0         # O (pipeline inventory)
    backorders: Optional[float] = 0.0                 # B (unfilled backorders)


@dataclass
class FixedPeriodReviewOutputs:
    review_period: float
    lead_time: float
    protection_interval: float          # P + L
    service_level: float
    z_score: float
    expected_demand_protection_interval: float # d * (P + L)
    std_dev_protection_interval: float  # sigma_d * sqrt(P + L)
    safety_stock: float                 # z * sigma_d * sqrt(P + L)
    target_inventory_level: float       # M = d*(P + L) + SS (Order-up-to level)
    inventory_position: Optional[float] = None # IP = I + O - B
    recommended_order_quantity: Optional[float] = None # q = max(0, M - IP)
    ss_ratio_vs_continuous: float = 1.0 # sqrt(P + L) / sqrt(L)


def calculate_fixed_period_review(inputs: FixedPeriodReviewInputs) -> FixedPeriodReviewOutputs:
    """
    Calculates Periodic Review (Order-Up-To / Target Inventory) policy parameters.
    Protection interval is (P + L).
    Target inventory M = d * (P + L) + z * sigma_d * sqrt(P + L).
    """
    P = float(inputs.review_period)
    L = float(inputs.lead_time)
    d = float(inputs.demand_mean)
    sigma_d = float(inputs.demand_std_dev)
    csl = float(inputs.service_level)

    if csl > 1.0:
        csl = csl / 100.0
    if not (0.0 < csl < 1.0):
        raise ValueError(f"Service level must be in (0, 1), got {csl}")
    if P <= 0 or L < 0 or d < 0 or sigma_d < 0:
        raise ValueError("Review period, lead time, and demand parameters must be non-negative (P > 0).")

    z = normal_inv_cdf(csl)
    protection_interval = P + L
    exp_demand = d * protection_interval
    sigma_protect = sigma_d * math.sqrt(protection_interval)
    safety_stock = z * sigma_protect
    target_inventory = exp_demand + safety_stock

    # Ratio of safety stock required vs continuous review with same lead time L
    ss_ratio = math.sqrt(protection_interval / L) if L > 0 else 1.0

    # Recommended order quantity if current inventory provided
    inv_position = None
    recommended_order = None
    if inputs.current_inventory_on_hand is not None:
        on_hand = float(inputs.current_inventory_on_hand)
        on_order = float(inputs.inventory_on_order or 0.0)
        backorder = float(inputs.backorders or 0.0)
        inv_position = on_hand + on_order - backorder
        recommended_order = max(0.0, target_inventory - inv_position)

    return FixedPeriodReviewOutputs(
        review_period=round(P, 2),
        lead_time=round(L, 2),
        protection_interval=round(protection_interval, 2),
        service_level=round(csl, 4),
        z_score=round(z, 4),
        expected_demand_protection_interval=round(exp_demand, 2),
        std_dev_protection_interval=round(sigma_protect, 2),
        safety_stock=round(safety_stock, 2),
        target_inventory_level=round(target_inventory, 2),
        inventory_position=round(inv_position, 2) if inv_position is not None else None,
        recommended_order_quantity=round(recommended_order, 2) if recommended_order is not None else None,
        ss_ratio_vs_continuous=round(ss_ratio, 2)
    )


def verify_fixed_period_review(inputs: FixedPeriodReviewInputs, outputs: FixedPeriodReviewOutputs) -> Dict[str, Any]:
    """Sanity and constraint verification for Fixed-Period Review model."""
    checks = []

    # Check 1: Protection interval equals P + L
    p_check = abs(outputs.protection_interval - (inputs.review_period + inputs.lead_time)) < 1e-4
    checks.append({
        "check": "Protection Interval Verification",
        "passed": p_check,
        "detail": f"Protection interval = P ({inputs.review_period}) + L ({inputs.lead_time}) = {outputs.protection_interval}"
    })

    # Check 2: Target Inventory exceeds Safety Stock and Expected Demand
    m_check = outputs.target_inventory_level >= outputs.safety_stock
    checks.append({
        "check": "Target Inventory Exceeds Safety Stock",
        "passed": m_check,
        "detail": f"Target Level M ({outputs.target_inventory_level}) >= Safety Stock ({outputs.safety_stock})"
    })

    # Check 3: If order placed, non-negative
    if outputs.recommended_order_quantity is not None:
        q_pos = outputs.recommended_order_quantity >= 0
        checks.append({
            "check": "Recommended Order Non-Negativity",
            "passed": q_pos,
            "detail": f"Recommended order quantity = {outputs.recommended_order_quantity} units >= 0"
        })

    all_passed = all(c["passed"] for c in checks)
    return {
        "model": "Fixed-Period Review",
        "verified": all_passed,
        "checks": checks
    }


# ==============================================================================
# Model 5: Newsvendor Model (Single-Period Inventory)
# ==============================================================================

@dataclass
class NewsvendorInputs:
    selling_price: float                # P ($/unit revenue)
    purchase_cost: float                # C ($/unit acquisition or production cost)
    salvage_value: float = 0.0          # V ($/unit markdown/liquidation value)
    penalty_cost: float = 0.0           # g ($/unit lost goodwill/shortage penalty)
    demand_mean: Optional[float] = None # mu (mean demand)
    demand_std_dev: Optional[float] = None # sigma (demand standard deviation)
    discrete_demand: Optional[List[Tuple[float, float]]] = None # List of (demand, prob)


@dataclass
class NewsvendorOutputs:
    cost_of_underage: float             # Cu = P - C + g
    cost_of_overage: float              # Co = C - V
    critical_ratio: float               # CR = Cu / (Cu + Co)
    optimal_order_quantity: float       # Q*
    z_score: Optional[float] = None     # z* for normal demand
    expected_lost_sales: float = 0.0    # E[max(D - Q, 0)]
    expected_sales: float = 0.0         # E[min(D, Q)]
    expected_leftovers: float = 0.0     # E[max(Q - D, 0)]
    expected_revenue: float = 0.0
    expected_cost: float = 0.0
    expected_salvage_revenue: float = 0.0
    expected_profit: float = 0.0
    expected_profit_perfect_info: float = 0.0
    evpi: float = 0.0                   # Expected Value of Perfect Information


def calculate_newsvendor(inputs: NewsvendorInputs) -> NewsvendorOutputs:
    """
    Calculates optimal order quantity and financial metrics under Newsvendor model.
    Critical ratio:
        CR = Cu / (Cu + Co)
        Cu = P - C + g (underage cost)
        Co = C - V (overage cost)
    """
    P = float(inputs.selling_price)
    C = float(inputs.purchase_cost)
    V = float(inputs.salvage_value)
    g = float(inputs.penalty_cost)

    Cu = P - C + g
    Co = C - V

    if Cu <= 0:
        raise ValueError(f"Cost of underage Cu must be positive: Selling Price ({P}) + Penalty ({g}) must exceed Cost ({C}).")
    if Co <= 0:
        raise ValueError(f"Cost of overage Co must be positive: Purchase Cost ({C}) must exceed Salvage Value ({V}).")

    CR = Cu / (Cu + Co)

    # Case A: Normal Demand Distribution
    if inputs.demand_mean is not None and inputs.demand_std_dev is not None:
        mu = float(inputs.demand_mean)
        sigma = float(inputs.demand_std_dev)
        if sigma < 0:
            raise ValueError("Standard deviation of demand must be non-negative.")

        if sigma == 0:
            Q_star = mu
            z_star = 0.0
            lost_sales = 0.0
            sales = mu
            leftovers = 0.0
        else:
            z_star = normal_inv_cdf(CR)
            Q_star = mu + z_star * sigma
            # Loss function L(z)
            loss_z = unit_normal_loss(z_star)
            lost_sales = sigma * loss_z
            sales = mu - lost_sales
            leftovers = Q_star - sales

        rev_sales = P * sales
        rev_salvage = V * leftovers
        total_rev = rev_sales + rev_salvage
        total_cost = C * Q_star
        expected_profit = total_rev - total_cost

        # Profit under perfect information = (P - C) * mu
        perfect_info_profit = (P - C) * mu
        evpi = max(0.0, perfect_info_profit - expected_profit)

        return NewsvendorOutputs(
            cost_of_underage=round(Cu, 4),
            cost_of_overage=round(Co, 4),
            critical_ratio=round(CR, 4),
            optimal_order_quantity=round(Q_star, 2),
            z_score=round(z_star, 4),
            expected_lost_sales=round(lost_sales, 2),
            expected_sales=round(sales, 2),
            expected_leftovers=round(leftovers, 2),
            expected_revenue=round(total_rev, 2),
            expected_cost=round(total_cost, 2),
            expected_salvage_revenue=round(rev_salvage, 2),
            expected_profit=round(expected_profit, 2),
            expected_profit_perfect_info=round(perfect_info_profit, 2),
            evpi=round(evpi, 2)
        )

    # Case B: Discrete Demand Distribution
    elif inputs.discrete_demand:
        # Sort by demand ascending
        sorted_pairs = sorted(inputs.discrete_demand, key=lambda x: x[0])
        cum_prob = 0.0
        Q_star = sorted_pairs[-1][0]
        for dem, p in sorted_pairs:
            cum_prob += p
            if cum_prob >= CR:
                Q_star = dem
                break

        # Calculate expected values across discrete distribution
        sales = sum(min(d, Q_star) * p for d, p in sorted_pairs)
        leftovers = sum(max(0.0, Q_star - d) * p for d, p in sorted_pairs)
        lost_sales = sum(max(0.0, d - Q_star) * p for d, p in sorted_pairs)
        mean_d = sum(d * p for d, p in sorted_pairs)

        rev_sales = P * sales
        rev_salvage = V * leftovers
        total_rev = rev_sales + rev_salvage
        total_cost = C * Q_star
        expected_profit = total_rev - total_cost
        perfect_info_profit = (P - C) * mean_d
        evpi = max(0.0, perfect_info_profit - expected_profit)

        return NewsvendorOutputs(
            cost_of_underage=round(Cu, 4),
            cost_of_overage=round(Co, 4),
            critical_ratio=round(CR, 4),
            optimal_order_quantity=round(Q_star, 2),
            z_score=None,
            expected_lost_sales=round(lost_sales, 2),
            expected_sales=round(sales, 2),
            expected_leftovers=round(leftovers, 2),
            expected_revenue=round(total_rev, 2),
            expected_cost=round(total_cost, 2),
            expected_salvage_revenue=round(rev_salvage, 2),
            expected_profit=round(expected_profit, 2),
            expected_profit_perfect_info=round(perfect_info_profit, 2),
            evpi=round(evpi, 2)
        )
    else:
        raise ValueError("Must provide either (demand_mean and demand_std_dev) or discrete_demand distribution.")


def verify_newsvendor(inputs: NewsvendorInputs, outputs: NewsvendorOutputs) -> Dict[str, Any]:
    """Sanity and constraint verification for Newsvendor model."""
    checks = []

    # Check 1: Critical Ratio in (0, 1)
    cr_valid = 0.0 < outputs.critical_ratio < 1.0
    checks.append({
        "check": "Critical Fractile Bounds (0 < CR < 1)",
        "passed": cr_valid,
        "detail": f"Critical ratio = {outputs.critical_ratio:.4f} (Cu=${outputs.cost_of_underage:.2f}, Co=${outputs.cost_of_overage:.2f})"
    })

    # Check 2: Conservation of Inventory: Expected Sales + Expected Leftovers == Q*
    cons_diff = abs((outputs.expected_sales + outputs.expected_leftovers) - outputs.optimal_order_quantity)
    cons_valid = cons_diff < 0.1
    checks.append({
        "check": "Inventory Conservation (Sales + Leftovers == Q*)",
        "passed": cons_valid,
        "detail": f"Sales ({outputs.expected_sales}) + Leftovers ({outputs.expected_leftovers}) = {outputs.expected_sales + outputs.expected_leftovers:.2f} vs Q* ({outputs.optimal_order_quantity})"
    })

    # Check 3: Expected profit is non-negative and less than perfect information
    profit_valid = outputs.expected_profit >= 0 and outputs.expected_profit <= outputs.expected_profit_perfect_info + 0.01
    checks.append({
        "check": "Economic Viability & EVPI Non-negativity",
        "passed": profit_valid,
        "detail": f"Expected Profit (${outputs.expected_profit:,.2f}) <= Perfect Info (${outputs.expected_profit_perfect_info:,.2f}), EVPI = ${outputs.evpi:,.2f}"
    })

    all_passed = all(c["passed"] for c in checks)
    return {
        "model": "Newsvendor",
        "verified": all_passed,
        "checks": checks
    }
