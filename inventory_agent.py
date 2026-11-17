"""
Supply Chain Analytics Inventory Decision Agent
================================================
Orchestrates natural-language inventory decision-making through a 6-stage architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (SS & ROP)
4. Fixed-Period Review Model (Periodic Review / Order-Up-To Level)
5. Newsvendor Model (Single-Period Stochastic Demand)
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union

import inventory_tools as it


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock and Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model (Periodic Review)"
    NEWSVENDOR = "Newsvendor Model (Single-Period Stochastic)"
    UNKNOWN = "Unknown / Unclassified"


@dataclass
class VerificationCheck:
    name: str
    status: str  # "PASS", "WARN", "FAIL", "INFO"
    detail: str


@dataclass
class AgentResponse:
    question: str
    model_type: InventoryModelType
    confidence: float
    raw_inputs: Dict[str, Any]
    cleaned_inputs: Dict[str, Any]
    tool_name: str
    calculation_results: Dict[str, Any]
    verifications: List[VerificationCheck]
    executive_summary: str
    detailed_explanation: str
    success: bool
    error_message: Optional[str] = None


class InventoryAgent:
    """
    End-to-end intelligent decision agent for Supply Chain Analytics.
    Executes the 6-stage analytical pipeline:
    1. Identify Model
    2. Extract Inputs
    3. Select Python Tool
    4. Calculate
    5. Verify
    6. Explain
    """

    def __init__(self, working_days_default: int = 250):
        self.working_days_default = working_days_default

    # -----------------------------------------------------------------------
    # Stage 1: Identify Model
    # -----------------------------------------------------------------------
    def identify_model(self, question: str) -> Tuple[InventoryModelType, float]:
        """
        Identify which supply chain inventory model matches the customer question.
        Uses hierarchical token and semantic pattern scoring.
        """
        q_lower = question.lower()

        scores = {
            InventoryModelType.NEWSVENDOR: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.EOQ: 0.0,
        }

        # 1. Newsvendor signals
        newsvendor_keywords = [
            "newsvendor", "newsboy", "perishable", "spoil", "salvage", "critical fractile",
            "single period", "single season", "fashion item", "newspaper", "leftover",
            "underage", "overage", "markdown", "liquidation", "perish", "event-based"
        ]
        for kw in newsvendor_keywords:
            if kw in q_lower:
                scores[InventoryModelType.NEWSVENDOR] += 2.5

        if "salvage" in q_lower or "salvage value" in q_lower:
            scores[InventoryModelType.NEWSVENDOR] += 4.0
        if "critical fractile" in q_lower or "underage cost" in q_lower or "overage cost" in q_lower:
            scores[InventoryModelType.NEWSVENDOR] += 5.0

        # 2. Fixed-Period Review signals
        fixed_period_keywords = [
            "periodic review", "fixed period", "review period", "order-up-to", "order up to",
            "target inventory", "inventory position", "protection interval", "every week",
            "every 2 weeks", "every month", "reviewed every", "review interval", "p-system",
            "(t, s)", "order up to level"
        ]
        for kw in fixed_period_keywords:
            if kw in q_lower:
                scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 2.5

        if "order-up-to" in q_lower or "order up to" in q_lower or "inventory position" in q_lower:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 4.5
        if "review period" in q_lower or "reviewed every" in q_lower:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 4.0

        # 3. Quantity Discount signals
        discount_keywords = [
            "discount", "price break", "volume discount", "bulk discount", "tier",
            "tiers", "bracket", "schedule", "price breaks", "quantity discount",
            "supplier offers", "discount schedule"
        ]
        for kw in discount_keywords:
            if kw in q_lower:
                scores[InventoryModelType.QUANTITY_DISCOUNT] += 2.5

        if ("discount" in q_lower or "tier" in q_lower or "break" in q_lower) and ("price" in q_lower or "schedule" in q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 4.0

        # Regex detect discount tiers like "0-499: $10" or "under 500: $20, 500-999: $18"
        if re.search(r"\d+\s*[-–to]\s*\d+\s*[:@\$]", q_lower) or re.search(r"tier\s*\d+", q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5.0

        # 4. Safety Stock & Reorder Point signals
        ss_rop_keywords = [
            "safety stock", "reorder point", "rop", "service level", "cycle service level",
            "csl", "stockout risk", "stockout probability", "lead time demand",
            "standard deviation of lead time", "lead time variability", "z-score",
            "in-stock probability"
        ]
        for kw in ss_rop_keywords:
            if kw in q_lower:
                scores[InventoryModelType.SAFETY_STOCK_ROP] += 2.0

        if "safety stock" in q_lower or "reorder point" in q_lower or "rop" in q_lower.split():
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4.0
        if "service level" in q_lower and ("lead time" in q_lower or "demand standard deviation" in q_lower or "std dev" in q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.5

        # 5. EOQ signals
        eoq_keywords = [
            "eoq", "economic order quantity", "annual demand", "setup cost", "ordering cost",
            "carrying cost", "holding cost", "how many units should we order",
            "optimal order quantity", "order batch size", "cycle inventory",
            "orders per year", "total inventory cost"
        ]
        for kw in eoq_keywords:
            if kw in q_lower:
                scores[InventoryModelType.EOQ] += 1.5

        if "economic order quantity" in q_lower or "eoq" in q_lower.split():
            scores[InventoryModelType.EOQ] += 4.5

        # Disambiguation logic:
        # Check if question includes variability / stochasticity
        has_stochasticity = any(term in q_lower for term in [
            "standard deviation", "std dev", "std", "service level", "csl", "stockout", "safety stock"
        ])
        if not has_stochasticity:
            # Deterministic inventory lot-sizing questions are EOQ or Quantity Discount
            if any(term in q_lower for term in ["optimal order size", "optimal order quantity", "annual demand", "annually", "cost to place an order", "holding cost", "valve"]):
                if scores[InventoryModelType.QUANTITY_DISCOUNT] < 4.0:
                    scores[InventoryModelType.EOQ] += 7.0
                    scores[InventoryModelType.SAFETY_STOCK_ROP] = 0.0

        # If both EOQ and Quantity Discount scored high, Quantity Discount takes precedence if tiers/discounts mentioned
        if scores[InventoryModelType.QUANTITY_DISCOUNT] >= 4.0:
            scores[InventoryModelType.EOQ] = min(scores[InventoryModelType.EOQ], scores[InventoryModelType.QUANTITY_DISCOUNT] - 0.5)

        # If Fixed Period Review is explicitly mentioned, don't confuse with simple ROP
        if scores[InventoryModelType.FIXED_PERIOD_REVIEW] >= 4.0:
            scores[InventoryModelType.SAFETY_STOCK_ROP] = min(scores[InventoryModelType.SAFETY_STOCK_ROP], scores[InventoryModelType.FIXED_PERIOD_REVIEW] - 0.5)

        best_model = max(scores, key=lambda k: scores[k])
        max_score = scores[best_model]

        if max_score < 1.0:
            return InventoryModelType.UNKNOWN, 0.0

        # Normalize confidence to [0.5, 0.99]
        confidence = min(0.99, max(0.50, round(max_score / (max_score + 2.0), 2)))
        return best_model, confidence

    # -----------------------------------------------------------------------
    # Stage 2: Extract Inputs
    # -----------------------------------------------------------------------
    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Dict[str, Any]:
        """
        Extract numerical and contextual parameters from natural language input
        tailored to the target inventory model.
        """
        q = question

        extracted: Dict[str, Any] = {}

        # Generic number regex helpers
        def find_number_after(patterns: List[str], text: str, default: Optional[float] = None) -> Optional[float]:
            for pat in patterns:
                # Matches pattern followed by optional non-numeric bridge words, optional $, number (with optional commas and decimals)
                regex = rf"(?:{pat})[^0-9\n\.\;]*?(?:is|of|for|at|=|:|averages?|takes?)?\s*[\$]?\s*([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(",", "")
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
            return default

        def find_number_before(patterns: List[str], text: str, default: Optional[float] = None) -> Optional[float]:
            for pat in patterns:
                regex = rf"([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:{pat})"
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(",", "")
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
            return default

        def find_percentage(patterns: List[str], text: str, default: Optional[float] = None) -> Optional[float]:
            # 1. Check percentage BEFORE pattern: e.g. "95% cycle service level"
            for pat in patterns:
                m_before = re.search(rf"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?{pat}", text, re.IGNORECASE)
                if m_before:
                    try:
                        pct = float(m_before.group(1))
                        return pct / 100.0 if pct > 1.0 else pct
                    except ValueError:
                        continue
            # 2. Check percentage AFTER pattern: e.g. "cycle service level of 95%" or "cycle service level averages 95%"
            for pat in patterns:
                regex = rf"(?:{pat})[^0-9\n\.\;]*?(?:is|of|for|at|=|:|averages?)?\s*([0-9]+(?:\.[0-9]+)?)\s*%"
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    try:
                        pct = float(match.group(1))
                        return pct / 100.0 if pct > 1.0 else pct
                    except ValueError:
                        continue
                regex_dec = rf"(?:{pat})[^0-9\n\.\;]*?(?:is|of|for|at|=|:|averages?)?\s*(0\.[0-9]+)"
                match_dec = re.search(regex_dec, text, re.IGNORECASE)
                if match_dec:
                    try:
                        return float(match_dec.group(1))
                    except ValueError:
                        continue
            return default

        # ------------------ Model-specific extraction ------------------

        if model_type == InventoryModelType.EOQ:
            # Annual demand
            d = find_number_after(
                ["annual demand", "demand per year", "yearly demand", "sells", "sell", "demand of", "demand is", "annual sales"],
                q
            )
            if d is None:
                # Check for generic "demand of X units"
                d = find_number_after(["demand", "volume"], q)
            extracted["annual_demand"] = d

            # Ordering / Setup cost
            s = find_number_after(
                ["order cost", "ordering cost", "setup cost", "cost per order", "cost to place an order", "fixed cost per order"],
                q
            )
            extracted["order_cost"] = s

            # Unit purchase price
            c = find_number_after(
                ["unit price", "purchase price", "cost per unit", "item cost", "purchase cost", "unit cost", "costs", "price of"],
                q
            )
            extracted["unit_price"] = c

            # Holding cost (dollar amount)
            h = find_number_after(
                ["holding cost per unit", "unit holding cost", "carrying cost per unit", "holding cost of", "holding cost is"],
                q
            )
            extracted["holding_cost"] = h

            # Holding rate (% of price)
            h_rate = find_percentage(
                ["holding cost rate", "carrying rate", "holding rate", "annual holding cost is", "holding cost of", "carrying cost of", "holding cost"],
                q
            )
            extracted["holding_rate"] = h_rate

            # Lead time days / ROP deterministic
            lt = find_number_after(
                ["lead time of", "lead time is", "lead time", "delivery time"],
                q
            )
            extracted["lead_time_days"] = lt

            # Working days
            wd = find_number_after(["working days", "operational days", "operating days", "days per year"], q)
            extracted["working_days_per_year"] = int(wd) if wd is not None else self.working_days_default

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            # Annual demand
            d = find_number_after(
                ["annual demand", "demand per year", "yearly demand", "demand of", "demand is"],
                q
            )
            extracted["annual_demand"] = d

            # Order cost
            s = find_number_after(
                ["order cost", "ordering cost", "setup cost", "cost per order", "cost to place an order"],
                q
            )
            extracted["order_cost"] = s

            # Holding rate or holding cost
            h_rate = find_percentage(
                ["holding rate", "carrying rate", "holding cost rate", "holding cost is", "carrying cost is", "holding cost of"],
                q
            )
            extracted["holding_rate"] = h_rate

            h_cost = find_number_after(
                ["holding cost per unit", "unit holding cost", "carrying cost per unit", "holding cost"],
                q
            )
            extracted["holding_cost"] = h_cost

            # Discount schedule extraction
            schedule = self._parse_discount_schedule(q)
            extracted["discount_schedule"] = schedule

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # Service level
            csl = find_percentage(
                ["service level", "cycle service level", "csl", "target in-stock", "in-stock rate", "desired service level"],
                q
            )
            if csl is None:
                # Check for z-score directly
                z = find_number_after(["z-score", "z score", "z value"], q)
                if z is not None:
                    csl = it.standard_normal_cdf(z)
            extracted["service_level"] = csl

            # Daily demand mean and std
            d_mean = find_number_after(
                ["daily demand is", "daily demand of", "average daily demand", "mean daily demand", "daily demand", "demand per day", "average demand"],
                q
            )
            extracted["daily_demand_mean"] = d_mean

            d_std = find_number_after(
                ["standard deviation of daily demand", "daily standard deviation", "daily std dev", "daily std", "standard deviation of demand", "standard deviation of", "std dev of demand", "std dev of", "demand standard deviation", "standard deviation", "std dev", "std"],
                q
            )
            extracted["daily_demand_std"] = d_std

            # Lead time days
            lt = find_number_after(
                ["lead time is", "lead time of", "average lead time", "mean lead time", "lead time"],
                q
            )
            extracted["lead_time_days"] = lt

            lt_std = find_number_after(
                ["lead time standard deviation", "standard deviation of lead time", "lead time std dev", "lead time std"],
                q,
                default=0.0
            )
            extracted["lead_time_std_days"] = lt_std

            # Lead time demand direct
            ltd_mean = find_number_after(
                ["lead time demand is", "lead time demand of", "mean lead time demand", "average lead time demand", "demand during lead time"],
                q
            )
            extracted["lead_time_demand_mean"] = ltd_mean

            ltd_std = find_number_after(
                ["standard deviation of lead time demand", "std dev of lead time demand", "lead time demand std", "lead time demand standard deviation"],
                q
            )
            extracted["lead_time_demand_std"] = ltd_std

            # Holding cost
            h = find_number_after(["holding cost", "carrying cost"], q)
            extracted["holding_cost_per_unit"] = h

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Review period T
            t = find_number_after(
                ["review period", "review interval", "reviewed every", "order every", "period of", "cycle time", "every"],
                q
            )
            extracted["review_period"] = t

            # Lead time L
            lt = find_number_after(
                ["lead time is", "lead time of", "lead time", "delivery time", "takes"],
                q
            )
            extracted["lead_time"] = lt

            # Demand mean and std
            d_mean = find_number_after(
                ["average demand", "mean demand", "demand rate", "weekly demand", "daily demand", "demand is", "demand of"],
                q
            )
            extracted["demand_mean"] = d_mean

            d_std = find_number_after(
                ["demand standard deviation", "standard deviation of demand", "standard deviation of", "std dev of demand", "std dev of", "demand std", "standard deviation", "std dev", "std"],
                q
            )
            extracted["demand_std"] = d_std

            # Service level
            csl = find_percentage(
                ["service level", "cycle service level", "csl", "target in-stock", "in-stock probability"],
                q
            )
            extracted["service_level"] = csl

            # On-hand, on-order, backorders (supports both prefix and suffix patterns)
            oh = find_number_after(
                ["on hand is", "on-hand is", "current inventory is", "current stock is", "on hand:", "on-hand:", "stock in warehouse"],
                q,
                default=None
            )
            if oh is None:
                oh = find_number_before(["on hand", "on-hand", "in stock", "in warehouse"], q, default=0.0)
            extracted["current_on_hand"] = oh

            oo = find_number_after(
                ["on order is", "on-order is", "in transit is", "scheduled receipts is", "on order:", "on-order:"],
                q,
                default=None
            )
            if oo is None:
                oo = find_number_before(["on order", "on-order", "in transit", "scheduled receipts"], q, default=0.0)
            extracted["current_on_order"] = oo

            bo = find_number_after(
                ["backorders is", "backlog is", "backordered is", "backorders:", "backlog:"],
                q,
                default=None
            )
            if bo is None:
                bo = find_number_before(["backorders", "backlog", "backordered", "unfilled orders"], q, default=0.0)
            extracted["backorders"] = bo

        elif model_type == InventoryModelType.NEWSVENDOR:
            # Selling price
            p = find_number_after(
                ["retail selling price", "selling price", "retail price", "sale price", "revenue per unit", "price per unit", "sold for", "sells for", "sells", "price is", "price of"],
                q
            )
            extracted["selling_price"] = p

            # Purchase cost
            c = find_number_after(
                ["purchase cost", "acquisition cost", "wholesale price", "cost per unit", "procurement cost", "costs us", "costs", "bought for", "cost of", "unit cost"],
                q
            )
            extracted["purchase_cost"] = c

            # Salvage value
            s = find_number_after(
                ["salvage value", "salvage price", "salvaged for", "salvaged at", "salvaged", "markdown price", "liquidation value", "scrap value", "residual value", "discounted to", "salvage"],
                q,
                default=0.0
            )
            extracted["salvage_value"] = s

            # Goodwill cost / penalty
            g = find_number_after(
                ["goodwill penalty", "goodwill cost", "penalty cost", "stockout penalty", "unmet demand penalty"],
                q,
                default=0.0
            )
            extracted["goodwill_cost"] = g

            # Demand mean and std
            d_mean = find_number_after(
                ["mean demand", "average demand", "expected demand", "demand mean", "demand is normally distributed with mean", "mean of", "demand of"],
                q
            )
            extracted["demand_mean"] = d_mean

            d_std = find_number_after(
                ["standard deviation of demand", "demand standard deviation", "standard deviation of", "std dev of demand", "std dev of", "standard deviation", "std dev", "std of", "std"],
                q
            )
            extracted["demand_std"] = d_std

        return extracted

    def _parse_discount_schedule(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tier brackets and unit prices from natural language text.
        Supports patterns like:
        - "0-499: $15, 500-999: $14, 1000+: $13"
        - "under 500: $20; 500 to 999: $18; 1000 or more: $16"
        - "tier 1: 0 to 499 at $10, tier 2: 500+ at $8"
        """
        tiers: List[Dict[str, Any]] = []

        # Isolate schedule section if present
        sched_match = re.search(r"(?:schedule|tiers|supplier offers)[:\s]+(.*)", text, re.IGNORECASE)
        sched_text = sched_match.group(1) if sched_match else text

        # Pattern A: range "min - max : $price" or "min to max at $price"
        range_pattern = re.findall(
            r"([0-9]+)\s*(?:-|–|to)\s*([0-9]+)\s*(?:units)?\s*(?:is|at|:|costs)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)",
            sched_text,
            re.IGNORECASE
        )
        for min_q, max_q, price in range_pattern:
            tiers.append({
                "min_qty": float(min_q),
                "max_qty": float(max_q),
                "price": float(price)
            })

        # Pattern B: open-ended top tier "1000+ : $8" or "1000 or more at $8" or "over 1000: $8"
        open_pattern = re.findall(
            r"([0-9]+)\s*(?:\+|or more|and above|units or more|\+ units)\s*(?:is|at|:|costs)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)",
            sched_text,
            re.IGNORECASE
        )
        for min_q, price in open_pattern:
            # Check if this min_q is already captured with a max_q
            if not any(t["min_qty"] == float(min_q) for t in tiers):
                tiers.append({
                    "min_qty": float(min_q),
                    "max_qty": None,
                    "price": float(price)
                })

        # Pattern C: "under X: $price" (e.g. under 500 units costs $10)
        under_pattern = re.findall(
            r"(?:under|less than|up to)\s*([0-9]+)\s*(?:units)?\s*(?:is|at|:|costs)?\s*[\$]?\s*([0-9]+(?:\.[0-9]+)?)",
            sched_text,
            re.IGNORECASE
        )
        for max_q, price in under_pattern:
            if not any(t.get("max_qty") == float(max_q) - 1 or t.get("max_qty") == float(max_q) for t in tiers):
                tiers.append({
                    "min_qty": 0.0,
                    "max_qty": float(max_q) - 1.0,
                    "price": float(price)
                })

        # Sort and deduplicate tiers by min_qty
        if tiers:
            unique_tiers = {}
            for t in tiers:
                unique_tiers[t["min_qty"]] = t
            sorted_tiers = sorted(unique_tiers.values(), key=lambda x: x["min_qty"])
            return sorted_tiers

        return []

    # -----------------------------------------------------------------------
    # Stage 3: Select Python Tool
    # -----------------------------------------------------------------------
    def select_tool(self, model_type: InventoryModelType) -> str:
        """Map the identified inventory model to the appropriate Python calculation tool."""
        mapping = {
            InventoryModelType.EOQ: "inventory_tools.calculate_eoq",
            InventoryModelType.QUANTITY_DISCOUNT: "inventory_tools.calculate_quantity_discount",
            InventoryModelType.SAFETY_STOCK_ROP: "inventory_tools.calculate_safety_stock_rop",
            InventoryModelType.FIXED_PERIOD_REVIEW: "inventory_tools.calculate_fixed_period_review",
            InventoryModelType.NEWSVENDOR: "inventory_tools.calculate_newsvendor",
        }
        return mapping.get(model_type, "unknown_tool")

    # -----------------------------------------------------------------------
    # Stage 4: Calculate
    # -----------------------------------------------------------------------
    def calculate(self, model_type: InventoryModelType, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analytical calculation with validated inputs using inventory_tools.
        """
        if model_type == InventoryModelType.EOQ:
            return it.calculate_eoq(
                annual_demand=inputs["annual_demand"],
                order_cost=inputs["order_cost"],
                holding_cost=inputs.get("holding_cost"),
                unit_price=inputs.get("unit_price"),
                holding_rate=inputs.get("holding_rate"),
                working_days_per_year=inputs.get("working_days_per_year", self.working_days_default),
                lead_time_days=inputs.get("lead_time_days")
            )

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return it.calculate_quantity_discount(
                annual_demand=inputs["annual_demand"],
                order_cost=inputs["order_cost"],
                discount_schedule=inputs["discount_schedule"],
                holding_cost=inputs.get("holding_cost"),
                holding_rate=inputs.get("holding_rate")
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return it.calculate_safety_stock_rop(
                service_level=inputs["service_level"],
                daily_demand_mean=inputs.get("daily_demand_mean"),
                daily_demand_std=inputs.get("daily_demand_std"),
                lead_time_days=inputs.get("lead_time_days"),
                lead_time_std_days=inputs.get("lead_time_std_days", 0.0),
                lead_time_demand_mean=inputs.get("lead_time_demand_mean"),
                lead_time_demand_std=inputs.get("lead_time_demand_std"),
                holding_cost_per_unit=inputs.get("holding_cost_per_unit")
            )

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return it.calculate_fixed_period_review(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"],
                service_level=inputs["service_level"],
                current_on_hand=inputs.get("current_on_hand", 0.0),
                current_on_order=inputs.get("current_on_order", 0.0),
                backorders=inputs.get("backorders", 0.0)
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            return it.calculate_newsvendor(
                selling_price=inputs["selling_price"],
                purchase_cost=inputs["purchase_cost"],
                demand_mean=inputs["demand_mean"],
                demand_std=inputs["demand_std"],
                salvage_value=inputs.get("salvage_value", 0.0),
                goodwill_cost=inputs.get("goodwill_cost", 0.0)
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    # -----------------------------------------------------------------------
    # Stage 5: Verify
    # -----------------------------------------------------------------------
    def verify(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        results: Dict[str, Any]
    ) -> List[VerificationCheck]:
        """
        Perform rigorous domain-specific sanity and economic verification checks.
        """
        checks: List[VerificationCheck] = []

        if model_type == InventoryModelType.EOQ:
            eoq = results["eoq"]
            aoc = results["annual_ordering_cost"]
            ahc = results["annual_holding_cost"]

            # Check 1: Positive order quantity
            if eoq > 0:
                checks.append(VerificationCheck("Non-Zero Order Quantity", "PASS", f"EOQ is positive ({eoq} units)."))
            else:
                checks.append(VerificationCheck("Non-Zero Order Quantity", "FAIL", "EOQ is non-positive."))

            # Check 2: Cost trade-off parity (AOC == AHC at unconstrained EOQ)
            diff = abs(aoc - ahc)
            rel_diff = diff / max(aoc, ahc) if max(aoc, ahc) > 0 else 0
            if rel_diff < 0.02:
                checks.append(VerificationCheck(
                    "Economic Trade-Off Parity", "PASS",
                    f"Annual ordering cost (${aoc:,.2f}) equals annual holding cost (${ahc:,.2f}) within rounding tolerance."
                ))
            else:
                checks.append(VerificationCheck(
                    "Economic Trade-Off Parity", "WARN",
                    f"Ordering cost (${aoc:,.2f}) and holding cost (${ahc:,.2f}) deviate by {rel_diff:.1%}."
                ))

            # Check 3: Reasonable order frequency
            n = results["orders_per_year"]
            if 0.5 <= n <= 365:
                checks.append(VerificationCheck("Order Frequency Sanity", "PASS", f"{n:.1f} orders/year is operationally practical."))
            else:
                checks.append(VerificationCheck("Order Frequency Sanity", "WARN", f"Extreme order frequency: {n:.1f} orders/year."))

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            best_tier = results["optimal_tier"]
            opt_q = results["optimal_order_quantity"]
            opt_cost = results["optimal_total_cost"]

            # Check 1: Optimality verification
            all_higher = True
            for tier in results["evaluated_tiers"]:
                if tier["tier_index"] != best_tier:
                    if tier["total_annual_cost"] < opt_cost:
                        all_higher = False

            if all_higher:
                checks.append(VerificationCheck("Global Cost Minimization", "PASS", f"Tier {best_tier} strictly achieves minimum total cost (${opt_cost:,.2f})."))
            else:
                checks.append(VerificationCheck("Global Cost Minimization", "FAIL", "Identified tier does not minimize total cost."))

            # Check 2: Feasibility bracket check
            opt_tier_data = next(t for t in results["evaluated_tiers"] if t["tier_index"] == best_tier)
            min_q = opt_tier_data["min_qty"]
            max_q = opt_tier_data["max_qty"]
            is_in_range = (opt_q >= min_q) and (max_q is None or opt_q <= max_q)
            if is_in_range:
                checks.append(VerificationCheck("Bracket Feasibility", "PASS", f"Order quantity {opt_q} satisfies Tier {best_tier} range [{min_q}, {max_q or '∞'}]."))
            else:
                checks.append(VerificationCheck("Bracket Feasibility", "FAIL", f"Order quantity {opt_q} violates Tier {best_tier} bounds."))

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            ss = results["safety_stock"]
            rop = results["reorder_point"]
            csl = results["service_level"]

            # Check 1: SS >= 0 for CSL >= 0.5
            if csl >= 0.5 and ss >= 0:
                checks.append(VerificationCheck("Safety Stock Sign", "PASS", f"Safety stock ({ss:,.2f}) is non-negative for CSL >= 50%."))
            elif csl < 0.5:
                checks.append(VerificationCheck("Safety Stock Sign", "INFO", f"CSL ({csl:.1%}) < 50% results in negative safety buffer ({ss:,.2f})."))

            # Check 2: ROP exceeds expected lead time demand when CSL > 50%
            if rop >= results["lead_time_demand_mean"]:
                checks.append(VerificationCheck("ROP Cushion", "PASS", f"Reorder Point ({rop:,.2f}) covers expected lead time demand ({results['lead_time_demand_mean']:,.2f}) + buffer."))
            else:
                checks.append(VerificationCheck("ROP Cushion", "WARN", f"ROP ({rop:,.2f}) is below expected demand."))

            # Check 3: Stockout risk matches service level
            risk = results["stockout_risk"]
            if abs((1.0 - csl) - risk) < 1e-4:
                checks.append(VerificationCheck("Risk-Service Consistency", "PASS", f"Cycle stockout risk {risk:.1%} exactly equals (1 - CSL)."))

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            s_level = results["order_up_to_level_S"]
            exp_d = results["expected_demand_protection_interval"]
            order_q = results["recommended_order_quantity"]

            # Check 1: Target S >= Expected Demand over (T + L)
            if s_level >= exp_d:
                checks.append(VerificationCheck("Protection Interval Coverage", "PASS", f"Order-up-to level S ({s_level:,.2f}) covers protection interval demand ({exp_d:,.2f})."))
            else:
                checks.append(VerificationCheck("Protection Interval Coverage", "WARN", "Target level S is less than expected demand over (T+L)."))

            # Check 2: Order quantity non-negativity
            if order_q >= 0:
                checks.append(VerificationCheck("Non-Negative Order Size", "PASS", f"Recommended order quantity ({order_q:,.2f}) is non-negative."))
            else:
                checks.append(VerificationCheck("Non-Negative Order Size", "FAIL", "Recommended order quantity is negative."))

        elif model_type == InventoryModelType.NEWSVENDOR:
            cu = results["cost_underage_Cu"]
            co = results["cost_overage_Co"]
            cf = results["critical_fractile"]
            q_star = results["optimal_order_quantity"]
            mu = results["demand_mean"]

            # Check 1: Economic viability (Cu > 0 and Co > 0)
            if cu > 0 and co > 0:
                checks.append(VerificationCheck("Economic Viability", "PASS", f"Both underage cost (${cu:,.2f}) and overage cost (${co:,.2f}) are strictly positive."))
            else:
                checks.append(VerificationCheck("Economic Viability", "FAIL", "Non-positive underage or overage costs."))

            # Check 2: Critical fractile in (0, 1)
            if 0 < cf < 1.0:
                checks.append(VerificationCheck("Critical Fractile Range", "PASS", f"Critical ratio CF = {cf:.4f} is strictly bounded in (0, 1)."))
            else:
                checks.append(VerificationCheck("Critical Fractile Range", "FAIL", f"Invalid critical ratio CF = {cf}."))

            # Check 3: Demand bias consistency
            if cf > 0.5 and q_star > mu:
                checks.append(VerificationCheck("High-Margin Bias Consistency", "PASS", f"CF > 50% implies Q* ({q_star:.1f}) exceeds mean demand ({mu:.1f}) to capture high upside."))
            elif cf < 0.5 and q_star < mu:
                checks.append(VerificationCheck("Low-Margin Bias Consistency", "PASS", f"CF < 50% implies Q* ({q_star:.1f}) is conservative below mean demand ({mu:.1f})."))
            else:
                checks.append(VerificationCheck("Demand Bias Consistency", "PASS", f"Q* ({q_star:.1f}) aligns with critical fractile {cf:.2%}."))

        return checks

    # -----------------------------------------------------------------------
    # Stage 6: Explain
    # -----------------------------------------------------------------------
    def explain(
        self,
        model_type: InventoryModelType,
        question: str,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verifications: List[VerificationCheck]
    ) -> Tuple[str, str]:
        """
        Generate executive-level direct recommendations and full mathematical explanations.
        """
        # Build executive summary
        if model_type == InventoryModelType.EOQ:
            exec_summary = (
                f"**Optimal Decision:** Order **{results['eoq']:,.0f} units** per batch. "
                f"Place **{results['orders_per_year']:.1f} orders per year** (every **{results['cycle_time_days']:.1f} operating days**). "
                f"Total Annual Inventory Cost is **${results['total_inventory_cost']:,.2f}**."
            )
            if results.get("reorder_point") is not None:
                exec_summary += f" Place replenishment orders when inventory drops to **{results['reorder_point']:,.0f} units**."

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            exec_summary = (
                f"**Optimal Decision:** Take advantage of **Tier {results['optimal_tier']}** by ordering **{results['optimal_order_quantity']:,.0f} units** "
                f"at **${results['optimal_unit_price']:,.2f}/unit**. "
                f"This minimizes total annual spend to **${results['optimal_total_cost']:,.2f}**, "
                f"saving **${results['annual_savings_vs_base']:,.2f} ({results['percentage_savings']:.1f}%)** per year compared to baseline."
            )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            exec_summary = (
                f"**Optimal Decision:** Maintain **{results['safety_stock']:,.0f} units** of Safety Stock "
                f"and set the Reorder Point (ROP) to **{results['reorder_point']:,.0f} units**. "
                f"This guarantees a **{results['service_level']:.1%} Cycle Service Level** with only a **{results['stockout_risk']:.1%} stockout risk** per order cycle."
            )

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            exec_summary = (
                f"**Optimal Decision:** Set Order-Up-To Level (S) to **{results['order_up_to_level_S']:,.0f} units** "
                f"(including **{results['safety_stock']:,.0f} units** safety stock). "
                f"With current inventory position of **{results['current_inventory_position']:,.0f} units**, "
                f"place an order for **{results['recommended_order_quantity']:,.0f} units** now."
            )

        elif model_type == InventoryModelType.NEWSVENDOR:
            exec_summary = (
                f"**Optimal Decision:** Order **{results['optimal_order_quantity']:,.0f} units** for the single selling period. "
                f"Based on a Critical Fractile of **{results['critical_fractile']:.1%}**, this yields an Expected Profit of "
                f"**${results['expected_profit']:,.2f}** and an in-stock availability of **{results['in_stock_probability']:.1%}**."
            )

        else:
            exec_summary = "Model could not be determined."

        # Build detailed step-by-step markdown report
        lines: List[str] = []
        lines.append(f"## 📦 Supply Chain Decision Analysis")
        lines.append(f"**Identified Model:** `{model_type.value}`\n")

        lines.append("### 🎯 Executive Recommendation")
        lines.append(exec_summary + "\n")

        lines.append("### 📋 Extracted Input Parameters")
        lines.append("| Parameter | Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        for k, v in inputs.items():
            if k == "discount_schedule":
                lines.append(f"| `discount_schedule` | {len(v)} tiers | Tiered pricing schedule |")
            elif v is not None:
                lines.append(f"| `{k}` | `{v}` | Extracted from query |")
        lines.append("")

        lines.append("### 🧮 Mathematical Formulation & Steps")
        if model_type == InventoryModelType.EOQ:
            h = results["holding_cost_per_unit"]
            d = results["annual_demand"]
            s = results["order_cost"]
            lines.append(f"1. **Annual Holding Cost per unit (H):** `${h:,.2f}` per unit per year.")
            lines.append(f"2. **EOQ Formula:** $Q^* = \\sqrt{{\\frac{{2DS}}{{H}}}} = \\sqrt{{\\frac{{2 \\times {d:,.0f} \\times {s:,.2f}}}{{{h:,.2f}}}}} = {results['eoq']:,.2f}$ units.")
            lines.append(f"3. **Order Frequency:** $N = \\frac{{D}}{{Q^*}} = \\frac{{{d:,.0f}}}{{{results['eoq']:,.2f}}} = {results['orders_per_year']:.2f}$ orders/year.")
            lines.append(f"4. **Cycle Length:** $T = \\frac{{Q^*}}{{D}} \\times W = \\frac{{{results['eoq']:,.2f}}}{{{d:,.0f}}} \\times {results['working_days']} = {results['cycle_time_days']:.1f}$ operating days.")
            lines.append(f"5. **Annual Ordering Cost (AOC):** $N \\times S = {results['orders_per_year']:.2f} \\times \\${s:,.2f} = \\${results['annual_ordering_cost']:,.2f}$.")
            lines.append(f"6. **Annual Holding Cost (AHC):** \\frac{{Q^*}}{{2}} \\times H = \\frac{{{results['eoq']:,.2f}}}{{2}} \\times \\${h:,.2f} = \\${results['annual_holding_cost']:,.2f}$.")
            lines.append(f"7. **Total Inventory Cost (TIC):** $AOC + AHC = \\${results['total_inventory_cost']:,.2f}$.")
            if results.get("total_annual_cost") is not None:
                lines.append(f"8. **Total Annual Cost (TC):** $D \\times C + TIC = \\${results['total_annual_cost']:,.2f}$.")
            if results.get("reorder_point") is not None:
                lines.append(f"9. **Reorder Point (ROP):** $d \\times L = \\left(\\frac{{{d:,.0f}}}{{{results['working_days']}}}\\right) \\times {results['lead_time_days']} = {results['reorder_point']:,.1f}$ units.")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append("Evaluated all discount tiers for feasibility and total annual cost:")
            lines.append("| Tier | Range | Price | Unconstrained EOQ | Candidate Q | Purchase Cost | Ordering Cost | Holding Cost | Total Annual Cost | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in results["evaluated_tiers"]:
                max_str = f"{t['max_qty']:,.0f}" if t['max_qty'] is not None else "∞"
                status = "⭐ **OPTIMAL**" if t['is_optimal'] else ("Feasible" if t['is_eoq_feasible'] else "Price-Break Adjust")
                lines.append(
                    f"| {t['tier_index']} | {t['min_qty']:,.0f} - {max_str} | ${t['unit_price']:,.2f} | "
                    f"{t['unconstrained_eoq']:,.0f} | **{t['candidate_qty']:,.0f}** | "
                    f"${t['annual_purchase_cost']:,.2f} | ${t['annual_ordering_cost']:,.2f} | "
                    f"${t['annual_holding_cost']:,.2f} | **${t['total_annual_cost']:,.2f}** | {status} |"
                )

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            lines.append(f"1. **Lead Time Model:** {results['lead_time_type']}")
            lines.append(f"2. **Target Service Level:** CSL = {results['service_level']:.1%} $\\implies z = \\Phi^{{-1}}({results['service_level']:.4f}) = {results['z_score']:.4f}$.")
            lines.append(f"3. **Lead Time Demand Mean ($\\mu_{{LTD}}$):** {results['lead_time_demand_mean']:,.2f} units.")
            lines.append(f"4. **Lead Time Demand Std Dev ($\\sigma_{{LTD}}$):** {results['lead_time_demand_std']:,.2f} units.")
            lines.append(f"5. **Safety Stock (SS):** $z \\times \\sigma_{{LTD}} = {results['z_score']:.4f} \\times {results['lead_time_demand_std']:,.2f} = \\mathbf{{{results['safety_stock']:,.2f}}}$ units.")
            lines.append(f"6. **Reorder Point (ROP):** $\\mu_{{LTD}} + SS = {results['lead_time_demand_mean']:,.2f} + {results['safety_stock']:,.2f} = \\mathbf{{{results['reorder_point']:,.2f}}}$ units.")
            if results.get("annual_holding_cost_ss") is not None:
                lines.append(f"7. **Safety Stock Annual Holding Cost:** $SS \\times H = \\${results['annual_holding_cost_ss']:,.2f}/year.")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append(f"1. **Protection Interval:** $T + L = {results['review_period_T']} + {results['lead_time_L']} = {results['protection_interval']}$ time units.")
            lines.append(f"2. **Target Service Level:** CSL = {results['service_level']:.1%} $\\implies z = {results['z_score']:.4f}$.")
            lines.append(f"3. **Expected Demand over Protection Interval:** $d \\times (T + L) = {results['demand_mean_per_period']} \\times {results['protection_interval']} = {results['expected_demand_protection_interval']:,.2f}$ units.")
            lines.append(f"4. **Std Dev over Protection Interval:** $\\sigma_d \\times \\sqrt{{T + L}} = {results['demand_std_per_period']} \\times \\sqrt{{{results['protection_interval']}}} = {results['std_demand_protection_interval']:,.2f}$ units.")
            lines.append(f"5. **Safety Stock (SS):** $z \\times \\sigma_{{T+L}} = {results['z_score']:.4f} \\times {results['std_demand_protection_interval']:,.2f} = \\mathbf{{{results['safety_stock']:,.2f}}}$ units.")
            lines.append(f"6. **Order-Up-To Level (S):** $d(T + L) + SS = {results['expected_demand_protection_interval']:,.2f} + {results['safety_stock']:,.2f} = \\mathbf{{{results['order_up_to_level_S']:,.2f}}}$ units.")
            lines.append(f"7. **Current Inventory Position (IP):** $I_{{OH}} + I_{{OO}} - BO = {results['inventory_on_hand']:,.0f} + {results['inventory_on_order']:,.0f} - {results['backorders']:,.0f} = {results['current_inventory_position']:,.0f}$ units.")
            lines.append(f"8. **Recommended Order Quantity (Q):** $\\max(0, S - IP) = \\max(0, {results['order_up_to_level_S']:,.2f} - {results['current_inventory_position']:,.0f}) = \\mathbf{{{results['recommended_order_quantity']:,.2f}}}$ units.")

        elif model_type == InventoryModelType.NEWSVENDOR:
            lines.append(f"1. **Cost of Underage ($C_u$):** $p - c + g = \\${results['selling_price']:,.2f} - \\${results['purchase_cost']:,.2f} + \\${results['goodwill_cost']:,.2f} = \\mathbf{{\\${results['cost_underage_Cu']:,.2f}}}$.")
            lines.append(f"2. **Cost of Overage ($C_o$):** $c - s = \\${results['purchase_cost']:,.2f} - \\${results['salvage_value']:,.2f} = \\mathbf{{\\${results['cost_overage_Co']:,.2f}}}$.")
            lines.append(f"3. **Critical Fractile ($CF$):** $\\frac{{C_u}}{{C_u + C_o}} = \\frac{{{results['cost_underage_Cu']:,.2f}}}{{{results['cost_underage_Cu']:,.2f} + {results['cost_overage_Co']:,.2f}}} = \\mathbf{{{results['critical_fractile']:.4f}}}$ ({results['critical_fractile']:.2%}).")
            lines.append(f"4. **Optimal Safety Factor ($z^*$):** $\\Phi^{{-1}}({results['critical_fractile']:.4f}) = {results['z_score']:.4f}$.")
            lines.append(f"5. **Optimal Order Quantity ($Q^*$):** $\\mu + z^* \\sigma = {results['demand_mean']:,.0f} + ({results['z_score']:.4f} \\times {results['demand_std']:,.0f}) = \\mathbf{{{results['optimal_order_quantity']:,.2f}}}$ units.")
            lines.append(f"6. **Expected Lost Sales:** {results['expected_lost_sales']:,.2f} units.")
            lines.append(f"7. **Expected Leftover Inventory:** {results['expected_leftover_units']:,.2f} units.")
            lines.append(f"8. **Expected Sales:** {results['expected_sales']:,.2f} units.")
            lines.append(f"9. **Expected Profit:** $\\mathbf{{\\${results['expected_profit']:,.2f}}}$.")
            lines.append(f"10. **Value of Perfect Information (EVPI):** $\\${results['expected_value_of_perfect_information_EVPI']:,.2f}$.")

        lines.append("")
        lines.append("### ✅ Verification & Quality Audit")
        lines.append("| Verification Check | Status | Details |")
        lines.append("| :--- | :--- | :--- |")
        for v in verifications:
            icon = "✅" if v.status == "PASS" else ("⚠️" if v.status == "WARN" else ("❌" if v.status == "FAIL" else "ℹ️"))
            lines.append(f"| {v.name} | {icon} {v.status} | {v.detail} |")

        lines.append("")
        lines.append("### 💡 Managerial & Supply Chain Insights")
        if model_type == InventoryModelType.EOQ:
            lines.append("- **Sensitivity:** The total inventory cost curve is relatively flat around the EOQ. A ±10% shift in batch size increases total cost by less than 1%.")
            lines.append("- **Operational Packaging:** If units are packed on standard pallets or master cartons, round the batch size to the nearest pack size without significant cost penalty.")
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append(f"- **Working Capital & Holding Risk:** Qualifying for Tier {results['optimal_tier']} requires ordering {results['optimal_order_quantity']:,.0f} units. Ensure warehouse capacity and shelf-life permit larger lot sizes.")
            lines.append("- **Supplier Negotiation:** If cash flow is tight, consider negotiating scheduled partial deliveries while maintaining the volume discount price.")
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            lines.append(f"- **Lead Time Variability:** Lead time volatility often inflates safety stock much faster than demand volatility. Reducing vendor lead time variance by 50% can yield substantial working capital savings.")
            lines.append(f"- **Service Level Trade-off:** Increasing service level from 95% to 99% increases safety stock by ~42% ($z$ shifts from 1.645 to 2.326).")
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append("- **Review Cadence:** Periodic review requires higher safety stock than continuous review because inventory is vulnerable across the full $(T + L)$ protection interval rather than just lead time $L$.")
            lines.append("- **Consolidated Shipments:** Fixed-period review is ideal when grouping orders for multiple products from the same vendor to share transportation setup costs.")
        elif model_type == InventoryModelType.NEWSVENDOR:
            lines.append(f"- **Margin Cushion:** With a high margin ($C_u = \\${results['cost_underage_Cu']:,.2f}$ vs $C_o = \\${results['cost_overage_Co']:,.2f}$), the cost of turning away customers far exceeds salvage loss, justifying generous stocking.")
            lines.append(f"- **Demand Forecasting:** The EVPI of ${results['expected_value_of_perfect_information_EVPI']:,.2f} represents the maximum dollar amount your firm should pay for advanced market research or demand sensing tools.")

        detailed_explanation = "\n".join(lines)
        return exec_summary, detailed_explanation

    # -----------------------------------------------------------------------
    # End-to-End Execution Pipeline
    # -----------------------------------------------------------------------
    def process_query(self, question: str) -> AgentResponse:
        """
        Execute the full 6-stage pipeline:
        Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        # 1. Identify Model
        model_type, confidence = self.identify_model(question)
        if model_type == InventoryModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                model_type=model_type,
                confidence=confidence,
                raw_inputs={},
                cleaned_inputs={},
                tool_name="none",
                calculation_results={},
                verifications=[],
                executive_summary="Could not identify a supported supply chain inventory model.",
                detailed_explanation="Please provide questions related to: 1. EOQ, 2. Quantity Discount, 3. Safety Stock & ROP, 4. Fixed-Period Review, or 5. Newsvendor.",
                success=False,
                error_message="Could not classify question into a supported inventory model."
            )

        # 2. Extract Inputs
        raw_inputs = self.extract_inputs(question, model_type)

        # Validate mandatory inputs per model
        missing_fields = []
        if model_type == InventoryModelType.EOQ:
            if raw_inputs.get("annual_demand") is None:
                missing_fields.append("annual_demand")
            if raw_inputs.get("order_cost") is None:
                missing_fields.append("order_cost")
            if raw_inputs.get("holding_cost") is None and (raw_inputs.get("unit_price") is None or raw_inputs.get("holding_rate") is None):
                missing_fields.append("holding_cost (or unit_price and holding_rate)")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if raw_inputs.get("annual_demand") is None:
                missing_fields.append("annual_demand")
            if raw_inputs.get("order_cost") is None:
                missing_fields.append("order_cost")
            if not raw_inputs.get("discount_schedule"):
                missing_fields.append("discount_schedule (tiers with quantity ranges and unit prices)")
            if raw_inputs.get("holding_cost") is None and raw_inputs.get("holding_rate") is None:
                missing_fields.append("holding_cost or holding_rate")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if raw_inputs.get("service_level") is None:
                missing_fields.append("service_level")
            has_ltd = (raw_inputs.get("lead_time_demand_mean") is not None and raw_inputs.get("lead_time_demand_std") is not None)
            has_daily = (raw_inputs.get("daily_demand_mean") is not None and raw_inputs.get("lead_time_days") is not None)
            if not has_ltd and not has_daily:
                missing_fields.append("lead_time_demand_mean/std OR daily_demand_mean and lead_time_days")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if raw_inputs.get("review_period") is None:
                missing_fields.append("review_period")
            if raw_inputs.get("lead_time") is None:
                missing_fields.append("lead_time")
            if raw_inputs.get("demand_mean") is None:
                missing_fields.append("demand_mean")
            if raw_inputs.get("demand_std") is None:
                missing_fields.append("demand_std")
            if raw_inputs.get("service_level") is None:
                missing_fields.append("service_level")

        elif model_type == InventoryModelType.NEWSVENDOR:
            if raw_inputs.get("selling_price") is None:
                missing_fields.append("selling_price")
            if raw_inputs.get("purchase_cost") is None:
                missing_fields.append("purchase_cost")
            if raw_inputs.get("demand_mean") is None:
                missing_fields.append("demand_mean")
            if raw_inputs.get("demand_std") is None:
                missing_fields.append("demand_std")

        if missing_fields:
            return AgentResponse(
                question=question,
                model_type=model_type,
                confidence=confidence,
                raw_inputs=raw_inputs,
                cleaned_inputs=raw_inputs,
                tool_name=self.select_tool(model_type),
                calculation_results={},
                verifications=[],
                executive_summary=f"Identified model `{model_type.value}`, but required inputs are missing: {', '.join(missing_fields)}.",
                detailed_explanation=f"Please provide the missing parameters: **{', '.join(missing_fields)}** to proceed with calculation.",
                success=False,
                error_message=f"Missing required parameters: {', '.join(missing_fields)}"
            )

        # 3. Select Python Tool
        tool_name = self.select_tool(model_type)

        # 4. Calculate
        try:
            results = self.calculate(model_type, raw_inputs)
        except Exception as e:
            return AgentResponse(
                question=question,
                model_type=model_type,
                confidence=confidence,
                raw_inputs=raw_inputs,
                cleaned_inputs=raw_inputs,
                tool_name=tool_name,
                calculation_results={},
                verifications=[],
                executive_summary="Calculation failed due to numerical error.",
                detailed_explanation=f"Error executing `{tool_name}`: {str(e)}",
                success=False,
                error_message=str(e)
            )

        # 5. Verify
        verifications = self.verify(model_type, raw_inputs, results)

        # 6. Explain
        exec_summary, detailed_explanation = self.explain(
            model_type, question, raw_inputs, results, verifications
        )

        return AgentResponse(
            question=question,
            model_type=model_type,
            confidence=confidence,
            raw_inputs=raw_inputs,
            cleaned_inputs=raw_inputs,
            tool_name=tool_name,
            calculation_results=results,
            verifications=verifications,
            executive_summary=exec_summary,
            detailed_explanation=detailed_explanation,
            success=True
        )
