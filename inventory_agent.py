"""
Supply Chain Analytics Inventory Decision Agent
===============================================
Implements the 6-step architecture:
  Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supports:
  1. Economic Order Quantity (EOQ)
  2. Quantity Discount Model
  3. Safety Stock & Reorder Point (Continuous Review)
  4. Fixed-Period Review Model (Order-Up-To)
  5. Newsvendor Model (Single-Period Inventory)

All numerical calculations are strictly executed by Python tools in inventory_tools.py.
"""

import os
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable

from inventory_tools import (
    EOQInputs,
    EOQOutputs,
    calculate_eoq,
    verify_eoq,
    DiscountTier,
    QuantityDiscountInputs,
    QuantityDiscountOutputs,
    calculate_quantity_discount,
    verify_quantity_discount,
    SafetyStockROPInputs,
    SafetyStockROPOutputs,
    calculate_safety_stock_and_rop,
    verify_safety_stock_rop,
    FixedPeriodReviewInputs,
    FixedPeriodReviewOutputs,
    calculate_fixed_period_review,
    verify_fixed_period_review,
    NewsvendorInputs,
    NewsvendorOutputs,
    calculate_newsvendor,
    verify_newsvendor,
)


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model (Periodic Review)"
    NEWSVENDOR = "Newsvendor Model (Single-Period)"
    UNKNOWN = "Unknown / Out of Scope"


@dataclass
class AgentStepTrace:
    step_number: int
    step_name: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecisionResult:
    question: str
    model_type: InventoryModelType
    extracted_inputs: Dict[str, Any]
    selected_tool_name: str
    raw_outputs: Any
    calculation_summary: Dict[str, Any]
    verification_report: Dict[str, Any]
    explanation: str
    step_traces: List[AgentStepTrace]
    success: bool
    error_message: Optional[str] = None


class InventoryDecisionAgent:
    """
    Autonomous Supply Chain Decision Agent that interprets natural language
    inventory questions and executes the 6-stage analytical pipeline.
    """

    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        self.use_llm = use_llm
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    # ==========================================================================
    # Step 1: Identify Model
    # ==========================================================================

    def identify_model(self, question: str) -> Tuple[InventoryModelType, str, float]:
        """
        Analyzes the customer question and routes it to the appropriate
        inventory decision model based on linguistic and domain signatures.
        Returns (model_type, rationale, confidence).
        """
        q = question.lower()

        # Score candidates based on specialized terminology
        scores: Dict[InventoryModelType, float] = {
            InventoryModelType.EOQ: 0.0,
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.NEWSVENDOR: 0.0,
        }

        # Model 2: Quantity Discount indicators
        if any(kw in q for kw in ["quantity discount", "price break", "volume discount", "discount tier", "price schedule", "price breaks", "all-units", "all units discount", "bracket"]):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 10.0
        if re.search(r"\d+\s*[-–to]+\s*\d+\s*(units)?\s*(at|@|costs?)\s*\$?\d+", q):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 8.0
        if re.search(r"(order|buy)\s*(at least|>=|>)\s*\d+", q):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 4.0

        # Model 5: Newsvendor indicators
        if any(kw in q for kw in ["newsvendor", "newsboy", "single period", "single-period", "perishable", "seasonal", "salvage value", "salvaged", "clearance price", "cost of underage", "cost of overage", "critical fractile", "critical ratio", "unsold units", "leftover units", "markdown", "fashion goods"]):
            scores[InventoryModelType.NEWSVENDOR] += 9.0
        if "salvage" in q or "unsold" in q or "leftover" in q:
            scores[InventoryModelType.NEWSVENDOR] += 5.0
        if "sell for" in q and "costs" in q and ("salvage" in q or "scrap" in q or "leftover" in q):
            scores[InventoryModelType.NEWSVENDOR] += 6.0

        # Model 4: Fixed-Period Review indicators
        if any(kw in q for kw in ["fixed period", "fixed-period", "periodic review", "order-up-to", "order up to", "target inventory", "review period", "review interval", "protection interval", "(r, s)", "(p, t)", "every p days"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 9.0
        if re.search(r"review\w*\s+every\s+\d+\s*(days|weeks|months)", q) or re.search(r"every\s+\d+\s*(days|weeks)\s+we\s+review", q):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 8.0
        if "on hand" in q and "on order" in q:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3.0

        # Model 3: Safety Stock & ROP indicators
        if any(kw in q for kw in ["safety stock", "reorder point", "rop", "cycle service level", "service level", "lead time demand", "stockout probability", "fill rate", "continuous review", "(s, q)"]):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 7.0
        if "lead time" in q and ("std dev" in q or "standard deviation" in q or "service level" in q or "confidence" in q):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 5.0

        # Model 1: EOQ indicators
        if any(kw in q for kw in ["eoq", "economic order quantity", "optimal order quantity", "how often should we order", "how many orders per year", "order cycle", "cycle inventory"]):
            scores[InventoryModelType.EOQ] += 7.0
        if "annual demand" in q and ("ordering cost" in q or "setup cost" in q or "holding cost" in q):
            scores[InventoryModelType.EOQ] += 6.0

        # If question has "safety stock" and "fixed period" / "review every", Model 4 wins
        if scores[InventoryModelType.FIXED_PERIOD_REVIEW] > 5.0 and scores[InventoryModelType.SAFETY_STOCK_ROP] > 0.0:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3.0

        # Determine winner
        best_model, highest_score = max(scores.items(), key=lambda item: item[1])

        if highest_score <= 1.0:
            return (
                InventoryModelType.UNKNOWN,
                "Unable to identify inventory model. The question did not contain enough recognizable inventory keywords.",
                0.0
            )

        confidence = min(0.99, 0.5 + (highest_score / 20.0))
        rationale = f"Matched key terminology and structural parameters for {best_model.value} (score: {highest_score:.1f})."
        return best_model, rationale, confidence

    # ==========================================================================
    # Step 2: Extract Inputs
    # ==========================================================================

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Dict[str, Any]:
        """
        Extracts numerical parameters, rates, time windows, and price schedules
        from the natural language query for the chosen inventory model.
        """
        extracted: Dict[str, Any] = {}
        q = question

        # Helper to extract numbers with regex
        def find_number(patterns: List[str], text: str) -> Optional[float]:
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace(",", "").replace("$", "").rstrip(".;,").strip()
                    try:
                        return float(val_str)
                    except ValueError:
                        continue
            return None

        # Helper to extract percentage or float in (0, 1)
        def find_percentage(patterns: List[str], text: str) -> Optional[float]:
            for pat in patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    val_str = match.group(1).replace("%", "").rstrip(".;,").strip()
                    try:
                        val = float(val_str)
                        return val / 100.0 if val > 1.0 else val
                    except ValueError:
                        continue
            return None

        if model_type == InventoryModelType.EOQ:
            # D: Annual Demand
            extracted["annual_demand"] = find_number([
                r"annual\s+demand\s*(?:of|is|=)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"demand\s*(?:of|is|=)?\s*[\$]?([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:per\s*year|annually|a\s*year)",
                r"sells?\s*[\$]?([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:per\s*year|annually)",
                r"demand\s*(?:of|is|=)?\s*[\$]?([\d,]+(?:\.\d+)?)",
            ], q)

            # S: Ordering / Setup Cost
            extracted["ordering_cost"] = find_number([
                r"(?:order|ordering|setup|procurement)\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:per|each)?\s*(?:order|setup)",
                r"costs?\s*[\$]([\d,]+(?:\.\d+)?)\s*to\s*(?:place|make)\s*(?:an?)?\s*order",
            ], q)

            # H: Holding Cost ($/unit/year)
            extracted["holding_cost"] = find_number([
                r"(?:annual\s+)?holding\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)\s*(?:per\s*unit|\/unit)?(?!\s*%)",
                r"carrying\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)\s*(?:per\s*unit|\/unit)?(?!\s*%)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:per\s*unit\s*per\s*year|\/unit\/year)",
            ], q)

            # h: Holding Cost Rate (%)
            extracted["holding_cost_rate"] = find_percentage([
                r"holding\s+cost\s+rate\s*(?:of|is|=|:)?\s*([\d\.]+)%",
                r"carrying\s+cost\s*(?:rate)?\s*(?:of|is|=|:)?\s*([\d\.]+)%",
                r"([\d\.]+)\s*%\s*(?:of\s*(?:the\s*)?(?:item|unit)?\s*(?:cost|price|value)|holding)",
            ], q)

            # C: Unit Cost
            extracted["unit_cost"] = find_number([
                r"unit\s+(?:cost|price)\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"costs?\s*[\$]([\d,]+(?:\.\d+)?)\s*(?:per|each|a)\s*unit",
                r"purchase\s+price\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:per\s*unit|each)",
            ], q)

            # L: Lead Time in Days
            extracted["lead_time_days"] = find_number([
                r"lead\s+time\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:working\s*)?days?",
                r"takes\s*([\d,]+(?:\.\d+)?)\s*days?\s*to\s*arrive",
            ], q)

            # W: Working Days per Year
            extracted["working_days_per_year"] = find_number([
                r"([\d,]+)\s*(?:working\s*)?days\s*(?:per\s*year|a\s*year)",
                r"operates?\s*([\d,]+)\s*days",
            ], q) or 250.0

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            # Demand & Ordering cost
            extracted["annual_demand"] = find_number([
                r"annual\s+demand\s*(?:of|is|=)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"demand\s*(?:of|is|=)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"sells?\s*[\$]?([\d,]+(?:\.\d+)?)",
            ], q)

            extracted["ordering_cost"] = find_number([
                r"(?:order|ordering|setup)\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:per|each)?\s*order",
            ], q)

            extracted["holding_cost_rate"] = find_percentage([
                r"holding\s+cost\s+(?:rate)?\s*(?:of|is|=|:)?\s*([\d\.]+)%",
                r"carrying\s+cost\s*(?:rate)?\s*(?:of|is|=|:)?\s*([\d\.]+)%",
                r"([\d\.]+)\s*%\s*(?:of\s*(?:the\s*)?(?:item|unit)?\s*(?:cost|price|value)|holding)",
            ], q)

            extracted["fixed_holding_cost"] = find_number([
                r"fixed\s+holding\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"holding\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)\s*(?:per\s*unit)",
            ], q)

            # Extract tiers
            # Patterns like: "1-499 units at $10, 500-999 units at $9, 1000+ units at $8"
            # or "0 to 999: $10, 1000 to 1999: $9.80, 2000+: $9.50"
            tiers: List[DiscountTier] = []
            
            # Pattern A: ranges like 1-999 at $10.00
            range_matches = list(re.finditer(
                r"([\d,]+)\s*(?:-|–|to)\s*([\d,]+)\s*(?:units)?\s*(?:at|@|costs?|is|:)?\s*[\$]?([\d\.]+)",
                q, re.IGNORECASE
            ))
            for m in range_matches:
                q_min = float(m.group(1).replace(",", ""))
                q_max = float(m.group(2).replace(",", ""))
                price = float(m.group(3).rstrip(".;,"))
                label = f"{int(q_min)}-{int(q_max)}"
                tiers.append(DiscountTier(min_quantity=q_min, max_quantity=q_max, unit_price=price, tier_label=label))

            # Pattern B: open ended tiers like "1000+ units at $8" or ">= 1000 at $8" or "1000 and above at $8"
            plus_matches = list(re.finditer(
                r"(?:(?:>=|>|at\s+least|more\s+than)\s*)?([\d,]+)\s*(?:\+|and\s+(?:above|over|more)|or\s+more)\s*(?:units)?\s*(?:at|@|costs?|is|:)?\s*[\$]?([\d\.]+)",
                q, re.IGNORECASE
            ))
            for m in plus_matches:
                q_min = float(m.group(1).replace(",", ""))
                price = float(m.group(2).rstrip(".;,"))
                label = f"{int(q_min)}+"
                tiers.append(DiscountTier(min_quantity=q_min, max_quantity=None, unit_price=price, tier_label=label))

            # If no tiers found with ranges, look for alternative patterns e.g. "tier 1: $10, tier 2: $9"
            if not tiers:
                # Default sample fallback tiers if question mentions tiers without full table
                alt_matches = list(re.finditer(r"(?:tier|bracket)\s*\d+.*?[\$]([\d\.]+)", q, re.IGNORECASE))
                if alt_matches:
                    for i, m in enumerate(alt_matches):
                        price = float(m.group(1).rstrip(".;,"))
                        tiers.append(DiscountTier(min_quantity=i * 500, max_quantity=(i+1)*500 - 1 if i < len(alt_matches)-1 else None, unit_price=price, tier_label=f"Tier {i+1}"))

            extracted["tiers"] = tiers

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            # Demand mean
            extracted["demand_mean"] = find_number([
                r"(?:average|mean|daily|weekly)?\s*demand\s*(?:mean|rate)?\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:units)?\s*(?:per\s*(?:day|week))?",
                r"sells?\s*([\d,]+(?:\.\d+)?)\s*units\s*(?:per\s*(?:day|week))?",
            ], q)

            # Demand standard deviation
            extracted["demand_std_dev"] = find_number([
                r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of\s*(?:demand)?)?\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"demand\s*(?:std\s*dev|standard\s+deviation)\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

            # Lead time
            extracted["lead_time"] = find_number([
                r"lead\s+time\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)?",
                r"takes\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)\s*to\s*arrive",
            ], q)

            # Lead time std dev
            extracted["lead_time_std_dev"] = find_number([
                r"lead\s+time\s*(?:std\s*dev|standard\s+deviation)\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"(?:std\s*dev|standard\s+deviation)\s*of\s*lead\s+time\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

            # Service level
            extracted["service_level"] = find_percentage([
                r"(?:cycle\s+)?service\s+level\s*(?:of|is|=|:)?\s*([\d\.]+)%?",
                r"fill\s+rate\s+target\s*(?:of|is|=|:)?\s*([\d\.]+)%?",
                r"([\d\.]+)%\s*(?:cycle\s+)?service\s+level",
                r"confidence\s*(?:level)?\s*(?:of|is|=|:)?\s*([\d\.]+)%?",
            ], q) or 0.95

            # Order quantity Q
            extracted["order_quantity"] = find_number([
                r"(?:order\s+quantity|batch\s+size|q)\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ], q)

            # Unit holding cost
            extracted["unit_holding_cost"] = find_number([
                r"holding\s+cost\s*(?:of|is|=|:)?\s*[\$]?([\d,]+(?:\.\d+)?)",
            ], q)

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Review period P
            extracted["review_period"] = find_number([
                r"review\s*[\w\s]{0,25}?\s*every\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)?",
                r"every\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)\s*(?:we\s*review|review)?",
                r"(?:review\s*(?:period|interval)?)\s*(?:is|=|of|every)?\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)?",
                r"order\s*every\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)",
            ], q)

            # Lead time L
            extracted["lead_time"] = find_number([
                r"([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?|months?)?\s*(?:replenishment\s*)?lead\s*time",
                r"lead\s+time\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)?",
                r"delivery\s+time\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"takes\s*([\d,]+(?:\.\d+)?)\s*(?:days?|weeks?)\s*to\s*arrive",
            ], q)

            # Demand mean & std dev
            extracted["demand_mean"] = find_number([
                r"(?:average|mean|daily|weekly)?\s*demand\s*(?:of|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"sells?\s*([\d,]+(?:\.\d+)?)\s*units",
                r"demand\s*(?:is|of|=)?\s*([\d,]+(?:\.\d+)?)",
            ], q)

            extracted["demand_std_dev"] = find_number([
                r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of\s*(?:demand)?|is|=|:)?\s*([\d,]+(?:\.\d+)?)",
                r"demand\s*(?:std\s*dev|standard\s+deviation)\s*(?:is|=|:)?\s*([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

            # Service level
            extracted["service_level"] = find_percentage([
                r"(?:service\s+level|csl)\s*(?:of|is|=|:)?\s*([\d\.]+)%?",
                r"([\d\.]+)%\s*service\s+level",
            ], q) or 0.95

            # Current inventory positions
            extracted["current_inventory_on_hand"] = find_number([
                r"(?:current\s*inventory|on\s*hand|inventory\s*on\s*hand)\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*units\s*on\s*hand",
            ], q)

            extracted["inventory_on_order"] = find_number([
                r"(?:on\s*order|pipeline|in\s*transit)\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
                r"([\d,]+(?:\.\d+)?)\s*units\s*on\s*order",
            ], q) or 0.0

            extracted["backorders"] = find_number([
                r"(?:backorders?|back\s*ordered)\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

        elif model_type == InventoryModelType.NEWSVENDOR:
            # Selling price P
            extracted["selling_price"] = find_number([
                r"(?:sell|sells|selling)\s+[\w\s]{0,35}?\s+for\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"(?:selling\s*price|retail\s*price|price|sells?\s*for)\s*(?:is|=|of)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:selling\s*price|per\s*unit\s*retail)",
            ], q)

            # Purchase cost C
            extracted["purchase_cost"] = find_number([
                r"(?:that\s+)?costs?\s*[\$]?([\d,]+(?:\.\d+)?)(?:\s*each|\s*to\s*buy|\s*to\s*make|\s*per\s*unit)?",
                r"(?:purchase\s*cost|unit\s*cost|acquisition\s*cost|costs?)\s*(?:is|=|of)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"[\$]([\d,]+(?:\.\d+)?)\s*(?:to\s*buy|each\s*cost)",
            ], q)

            # Salvage value V
            extracted["salvage_value"] = find_number([
                r"(?:unsold|leftover)\s*[\w\s]{0,30}?\s*(?:can\s*be\s*salvaged|salvaged|sold)\s*for\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"(?:salvage\s*value|clearance\s*price|salvaged\s*for|scrap\s*value)\s*(?:is|=|of)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"(?:unsold|leftover)\s*(?:units)?\s*(?:are|can\s*be)?\s*(?:sold|salvaged)\s*(?:for)?\s*[\$]?([\d,]+(?:\.\d+)?)",
                r"markdown\s*price\s*(?:is|=|of)?\s*[\$]?([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

            # Shortage penalty g
            extracted["penalty_cost"] = find_number([
                r"(?:penalty|shortage\s*penalty|goodwill\s*cost)\s*(?:is|=|of)?\s*[\$]?([\d,]+(?:\.\d+)?)",
            ], q) or 0.0

            # Demand mean mu
            extracted["demand_mean"] = find_number([
                r"(?:mean|average|expected)\s*(?:seasonal\s*)?demand\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
                r"demand\s*(?:is\s*normally\s*distributed\s*with\s*(?:a\s*)?)?mean\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
                r"mean\s*([\d,]+(?:\.\d+)?)",
                r"demand\s*of\s*([\d,]+(?:\.\d+)?)\s*(?:units)?",
            ], q)

            # Demand std dev sigma
            extracted["demand_std_dev"] = find_number([
                r"(?:standard\s*deviation|std\s*dev|sigma)\s*(?:is|=|of)?\s*([\d,]+(?:\.\d+)?)",
            ], q)

        return extracted

    # ==========================================================================
    # Step 3: Select Python Tool
    # ==========================================================================

    def select_tool(self, model_type: InventoryModelType) -> Tuple[str, Callable, str]:
        """
        Maps the identified model to its pure Python numerical calculation engine.
        Returns (tool_name, tool_callable, module_path).
        """
        mapping = {
            InventoryModelType.EOQ: (
                "calculate_eoq",
                calculate_eoq,
                "inventory_tools.calculate_eoq"
            ),
            InventoryModelType.QUANTITY_DISCOUNT: (
                "calculate_quantity_discount",
                calculate_quantity_discount,
                "inventory_tools.calculate_quantity_discount"
            ),
            InventoryModelType.SAFETY_STOCK_ROP: (
                "calculate_safety_stock_and_rop",
                calculate_safety_stock_and_rop,
                "inventory_tools.calculate_safety_stock_and_rop"
            ),
            InventoryModelType.FIXED_PERIOD_REVIEW: (
                "calculate_fixed_period_review",
                calculate_fixed_period_review,
                "inventory_tools.calculate_fixed_period_review"
            ),
            InventoryModelType.NEWSVENDOR: (
                "calculate_newsvendor",
                calculate_newsvendor,
                "inventory_tools.calculate_newsvendor"
            ),
        }
        if model_type not in mapping:
            raise ValueError(f"No registered calculation tool for model: {model_type}")
        return mapping[model_type]

    # ==========================================================================
    # Step 4: Calculate
    # ==========================================================================

    def calculate(self, model_type: InventoryModelType, inputs_dict: Dict[str, Any]) -> Any:
        """
        Executes the chosen Python tool using the extracted inputs.
        """
        tool_name, tool_func, _ = self.select_tool(model_type)

        if model_type == InventoryModelType.EOQ:
            if not inputs_dict.get("annual_demand") or not inputs_dict.get("ordering_cost"):
                raise ValueError("EOQ requires 'annual_demand' and 'ordering_cost'.")
            if not inputs_dict.get("holding_cost") and not (inputs_dict.get("holding_cost_rate") and inputs_dict.get("unit_cost")):
                raise ValueError("EOQ requires either 'holding_cost' or ('holding_cost_rate' and 'unit_cost').")

            typed_inputs = EOQInputs(
                annual_demand=inputs_dict["annual_demand"],
                ordering_cost=inputs_dict["ordering_cost"],
                holding_cost=inputs_dict.get("holding_cost"),
                holding_cost_rate=inputs_dict.get("holding_cost_rate"),
                unit_cost=inputs_dict.get("unit_cost"),
                lead_time_days=inputs_dict.get("lead_time_days"),
                working_days_per_year=inputs_dict.get("working_days_per_year", 250.0)
            )
            return tool_func(typed_inputs)

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if not inputs_dict.get("annual_demand") or not inputs_dict.get("ordering_cost"):
                raise ValueError("Quantity Discount requires 'annual_demand' and 'ordering_cost'.")
            if not inputs_dict.get("tiers"):
                raise ValueError("Quantity Discount requires at least one price tier/bracket.")

            typed_inputs = QuantityDiscountInputs(
                annual_demand=inputs_dict["annual_demand"],
                ordering_cost=inputs_dict["ordering_cost"],
                tiers=inputs_dict["tiers"],
                holding_cost_rate=inputs_dict.get("holding_cost_rate"),
                fixed_holding_cost=inputs_dict.get("fixed_holding_cost")
            )
            return tool_func(typed_inputs)

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if inputs_dict.get("lead_time") is None or inputs_dict.get("demand_mean") is None:
                raise ValueError("Safety Stock & ROP requires 'lead_time' and 'demand_mean'.")

            typed_inputs = SafetyStockROPInputs(
                lead_time=inputs_dict["lead_time"],
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict.get("demand_std_dev", 0.0),
                lead_time_std_dev=inputs_dict.get("lead_time_std_dev", 0.0),
                service_level=inputs_dict.get("service_level", 0.95),
                order_quantity=inputs_dict.get("order_quantity"),
                unit_holding_cost=inputs_dict.get("unit_holding_cost")
            )
            return tool_func(typed_inputs)

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if inputs_dict.get("review_period") is None or inputs_dict.get("lead_time") is None or inputs_dict.get("demand_mean") is None:
                raise ValueError("Fixed-Period Review requires 'review_period', 'lead_time', and 'demand_mean'.")

            typed_inputs = FixedPeriodReviewInputs(
                review_period=inputs_dict["review_period"],
                lead_time=inputs_dict["lead_time"],
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict.get("demand_std_dev", 0.0),
                service_level=inputs_dict.get("service_level", 0.95),
                current_inventory_on_hand=inputs_dict.get("current_inventory_on_hand"),
                inventory_on_order=inputs_dict.get("inventory_on_order", 0.0),
                backorders=inputs_dict.get("backorders", 0.0)
            )
            return tool_func(typed_inputs)

        elif model_type == InventoryModelType.NEWSVENDOR:
            if inputs_dict.get("selling_price") is None or inputs_dict.get("purchase_cost") is None:
                raise ValueError("Newsvendor requires 'selling_price' and 'purchase_cost'.")
            if inputs_dict.get("demand_mean") is None or inputs_dict.get("demand_std_dev") is None:
                raise ValueError("Newsvendor requires 'demand_mean' and 'demand_std_dev' (or discrete distribution).")

            typed_inputs = NewsvendorInputs(
                selling_price=inputs_dict["selling_price"],
                purchase_cost=inputs_dict["purchase_cost"],
                salvage_value=inputs_dict.get("salvage_value", 0.0),
                penalty_cost=inputs_dict.get("penalty_cost", 0.0),
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict["demand_std_dev"]
            )
            return tool_func(typed_inputs)

    # ==========================================================================
    # Step 5: Verify
    # ==========================================================================

    def verify(self, model_type: InventoryModelType, inputs_dict: Dict[str, Any], outputs: Any) -> Dict[str, Any]:
        """
        Executes formal mathematical constraint and domain sanity verifications.
        """
        if model_type == InventoryModelType.EOQ:
            typed_inputs = EOQInputs(
                annual_demand=inputs_dict["annual_demand"],
                ordering_cost=inputs_dict["ordering_cost"],
                holding_cost=inputs_dict.get("holding_cost"),
                holding_cost_rate=inputs_dict.get("holding_cost_rate"),
                unit_cost=inputs_dict.get("unit_cost"),
                lead_time_days=inputs_dict.get("lead_time_days"),
                working_days_per_year=inputs_dict.get("working_days_per_year", 250.0)
            )
            return verify_eoq(typed_inputs, outputs)

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            typed_inputs = QuantityDiscountInputs(
                annual_demand=inputs_dict["annual_demand"],
                ordering_cost=inputs_dict["ordering_cost"],
                tiers=inputs_dict["tiers"],
                holding_cost_rate=inputs_dict.get("holding_cost_rate"),
                fixed_holding_cost=inputs_dict.get("fixed_holding_cost")
            )
            return verify_quantity_discount(typed_inputs, outputs)

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            typed_inputs = SafetyStockROPInputs(
                lead_time=inputs_dict["lead_time"],
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict.get("demand_std_dev", 0.0),
                lead_time_std_dev=inputs_dict.get("lead_time_std_dev", 0.0),
                service_level=inputs_dict.get("service_level", 0.95),
                order_quantity=inputs_dict.get("order_quantity"),
                unit_holding_cost=inputs_dict.get("unit_holding_cost")
            )
            return verify_safety_stock_rop(typed_inputs, outputs)

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            typed_inputs = FixedPeriodReviewInputs(
                review_period=inputs_dict["review_period"],
                lead_time=inputs_dict["lead_time"],
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict.get("demand_std_dev", 0.0),
                service_level=inputs_dict.get("service_level", 0.95),
                current_inventory_on_hand=inputs_dict.get("current_inventory_on_hand"),
                inventory_on_order=inputs_dict.get("inventory_on_order", 0.0),
                backorders=inputs_dict.get("backorders", 0.0)
            )
            return verify_fixed_period_review(typed_inputs, outputs)

        elif model_type == InventoryModelType.NEWSVENDOR:
            typed_inputs = NewsvendorInputs(
                selling_price=inputs_dict["selling_price"],
                purchase_cost=inputs_dict["purchase_cost"],
                salvage_value=inputs_dict.get("salvage_value", 0.0),
                penalty_cost=inputs_dict.get("penalty_cost", 0.0),
                demand_mean=inputs_dict["demand_mean"],
                demand_std_dev=inputs_dict["demand_std_dev"]
            )
            return verify_newsvendor(typed_inputs, outputs)

        return {"verified": True, "checks": []}

    # ==========================================================================
    # Step 6: Explain
    # ==========================================================================

    def explain(
        self,
        question: str,
        model_type: InventoryModelType,
        inputs_dict: Dict[str, Any],
        outputs: Any,
        verification: Dict[str, Any]
    ) -> str:
        """
        Synthesizes the findings into a clear, structured supply chain decision report.
        Includes:
          1. Executive Decision Summary
          2. Model Selection & Context
          3. Extracted Parameters Table
          4. Mathematical Formulas & Step-by-Step Calculation
          5. Verification Summary
          6. Managerial Insights & Sensitivity Analysis
        """
        lines = []

        # Header
        lines.append(f"## 📦 Supply Chain Analytics Decision Report")
        lines.append(f"**Identified Model:** {model_type.value}")
        lines.append("")

        # 1. Executive Summary
        lines.append("### 🎯 1. Executive Decision Summary")
        if model_type == InventoryModelType.EOQ:
            lines.append(f"- **Recommended Order Quantity (Q*):** `{outputs.optimal_order_quantity:,.2f} units`")
            lines.append(f"- **Annual Order Frequency:** `{outputs.annual_orders_count:.2f} orders/year` (Order every `{outputs.order_cycle_days:.1f} working days`)")
            lines.append(f"- **Total Annual Relevant Cost (TRC):** `${outputs.total_relevant_cost:,.2f}` (Ordering: `${outputs.annual_ordering_cost:,.2f}`, Holding: `${outputs.annual_holding_cost:,.2f}`)")
            if outputs.reorder_point is not None:
                lines.append(f"- **Reorder Point (ROP):** Place a replenishment order when inventory hits `{outputs.reorder_point:,.2f} units`")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append(f"- **Optimal Purchasing Strategy:** Order `{outputs.optimal_order_quantity:,.0f} units` under **{outputs.optimal_tier_label}** at `${outputs.optimal_unit_price:,.2f}/unit`")
            lines.append(f"- **Total Annual Cost (Acquisition + Ordering + Holding):** `${outputs.min_total_annual_cost:,.2f}`")
            lines.append(f"- **Annual Savings vs Baseline:** `${outputs.annual_cost_savings:,.2f}` (`{outputs.percentage_savings:.1f}% net cost reduction`)")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            lines.append(f"- **Recommended Safety Stock (SS):** `{outputs.safety_stock:,.2f} units`")
            lines.append(f"- **Reorder Point (ROP):** Reorder when inventory drops to `{outputs.reorder_point:,.2f} units`")
            lines.append(f"- **Cycle Service Level:** `{outputs.service_level * 100:.1f}%` (Stockout risk per cycle: `{outputs.stockout_probability_per_cycle * 100:.1f}%`)")
            if outputs.item_fill_rate is not None:
                lines.append(f"- **Item Fill Rate (Type 2 Service Level):** `{outputs.item_fill_rate * 100:.2f}%`")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append(f"- **Target Order-Up-To Level (M):** `{outputs.target_inventory_level:,.2f} units`")
            lines.append(f"- **Safety Stock Requirement:** `{outputs.safety_stock:,.2f} units` (Protection interval: `{outputs.protection_interval:.0f} days`)")
            if outputs.recommended_order_quantity is not None:
                lines.append(f"- **Immediate Order Quantity:** Order `{outputs.recommended_order_quantity:,.2f} units` (Current inventory position: `{outputs.inventory_position:,.0f} units`)")

        elif model_type == InventoryModelType.NEWSVENDOR:
            lines.append(f"- **Optimal Order Quantity (Q*):** `{outputs.optimal_order_quantity:,.0f} units`")
            lines.append(f"- **Critical Fractile / Service Level:** `{outputs.critical_ratio * 100:.2f}%` (Cost of Underage: `${outputs.cost_of_underage:,.2f}`, Overage: `${outputs.cost_of_overage:,.2f}`)")
            lines.append(f"- **Expected Profit:** `${outputs.expected_profit:,.2f}` (Expected Sales: `{outputs.expected_sales:,.1f}`, Expected Leftovers: `{outputs.expected_leftovers:,.1f}`)")
            lines.append(f"- **Expected Value of Perfect Information (EVPI):** `${outputs.evpi:,.2f}`")

        lines.append("")

        # 2. Extracted Parameters Table
        lines.append("### 📋 2. Extracted Input Parameters")
        lines.append("| Parameter | Symbol | Value | Unit / Note |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for key, val in inputs_dict.items():
            if key == "tiers":
                lines.append(f"| Discount Schedule | tiers | {len(val)} tiers defined | Brackets evaluated |")
            elif val is not None:
                lines.append(f"| `{key.replace('_', ' ').title()}` | `{key}` | `{val}` | Input extracted from query |")
        lines.append("")

        # 3. Mathematical Formulas & Step-by-Step Calculation
        lines.append("### 📐 3. Mathematical Formulations & Calculation Steps")
        if model_type == InventoryModelType.EOQ:
            lines.append("#### Governing Formulas:")
            lines.append(r"$$Q^* = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$")
            lines.append(r"$$TRC = AOC + AHC = \frac{D}{Q^*} \cdot S + \frac{Q^*}{2} \cdot H$$")
            lines.append(r"$$N = \frac{D}{Q^*}, \quad T = \frac{Q^*}{D} \cdot W$$")
            lines.append("#### Step-by-Step Numerical Substitution:")
            D = inputs_dict["annual_demand"]
            S = inputs_dict["ordering_cost"]
            H = outputs.unit_holding_cost
            lines.append(f"1. **Annual Holding Cost per Unit ($H$):** `${H:.4f}`")
            lines.append(f"2. **Optimal Order Quantity ($Q^*$):**")
            lines.append(f"   $$Q^* = \\sqrt{{\\frac{{2 \\cdot {D:,.0f} \\cdot {S:,.2f}}}{{{H:.4f}}}}} = \\sqrt{{{ (2*D*S)/H:,.2f} }} = {outputs.optimal_order_quantity:,.2f}\\text{{ units}}$$")
            lines.append(f"3. **Annual Ordering Cost ($AOC$):** $({D:,.0f} / {outputs.optimal_order_quantity:,.2f}) \\cdot ${S:.2f} = ${outputs.annual_ordering_cost:,.2f}")
            lines.append(f"4. **Annual Holding Cost ($AHC$):** $({outputs.optimal_order_quantity:,.2f} / 2) \\cdot ${H:.4f} = ${outputs.annual_holding_cost:,.2f}")
            lines.append(f"5. **Order Cycle ($T$):** $({outputs.optimal_order_quantity:,.2f} / {D:,.0f}) \\cdot {inputs_dict.get('working_days_per_year', 250)} = {outputs.order_cycle_days:.1f}\\text{{ working days}}$")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append("#### Evaluated Tier Schedule:")
            lines.append("| Tier | Bracket | Price ($) | Raw EOQ | Feasible Q | Purchase Cost | Ordering Cost | Holding Cost | Total Annual Cost | Status |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for t in outputs.tier_evaluations:
                q_range = f"{int(t.min_quantity)}-{int(t.max_quantity) if t.max_quantity else '+'}"
                is_opt = " ⭐ **OPTIMAL**" if t.tier_index == outputs.optimal_tier_index else ("✓ Feasible" if t.is_eoq_feasible else "Adjusted to Min")
                lines.append(f"| {t.tier_label} | {q_range} | ${t.unit_price:.2f} | {t.unconstrained_eoq:,.1f} | {t.evaluated_quantity:,.1f} | ${t.annual_purchase_cost:,.2f} | ${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | **${t.total_annual_cost:,.2f}** | {is_opt} |")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            lines.append("#### Governing Formulas:")
            lines.append(r"$$\sigma_{LTD} = \sigma_d \sqrt{L} \quad (\text{or } \sqrt{L\sigma_d^2 + d^2\sigma_L^2})$$")
            lines.append(r"$$SS = z \cdot \sigma_{LTD}, \quad ROP = \mu_{LTD} + SS = (d \cdot L) + z \cdot \sigma_{LTD}$$")
            lines.append("#### Step-by-Step Substitution:")
            lines.append(f"1. **Cycle Service Level ($CSL$):** `{outputs.service_level * 100:.1f}%` $\\implies z = \\Phi^{{-1}}({outputs.service_level:.2f}) = {outputs.z_score:.4f}$")
            lines.append(f"2. **Expected Lead Time Demand ($\\mu_{{LTD}}$):** ${inputs_dict['demand_mean']} \\times {inputs_dict['lead_time']} = {outputs.expected_lead_time_demand:.2f}\\text{{ units}}$")
            lines.append(f"3. **Lead Time Demand Std Dev ($\\sigma_{{LTD}}$):** `{outputs.lead_time_demand_std_dev:.2f} units`")
            lines.append(f"4. **Safety Stock ($SS$):** ${outputs.z_score:.4f} \\times {outputs.lead_time_demand_std_dev:.2f} = {outputs.safety_stock:,.2f}\\text{{ units}}$")
            lines.append(f"5. **Reorder Point ($ROP$):** ${outputs.expected_lead_time_demand:.2f} + {outputs.safety_stock:.2f} = {outputs.reorder_point:,.2f}\\text{{ units}}$")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append("#### Governing Formulas:")
            lines.append(r"$$T_{\text{protect}} = P + L, \quad \sigma_{P+L} = \sigma_d \sqrt{P+L}$$")
            lines.append(r"$$M = d \cdot (P+L) + z \cdot \sigma_d \sqrt{P+L}, \quad q = \max(0, M - IP)$$")
            lines.append("#### Step-by-Step Substitution:")
            lines.append(f"1. **Protection Interval:** $P ({outputs.review_period}) + L ({outputs.lead_time}) = {outputs.protection_interval:.0f}\\text{{ periods}}$")
            lines.append(f"2. **Demand Uncertainty over Protection Interval ($\\sigma_{{P+L}}$):** ${inputs_dict['demand_std_dev']} \\times \\sqrt{{{outputs.protection_interval:.0f}}} = {outputs.std_dev_protection_interval:.2f}\\text{{ units}}$")
            lines.append(f"3. **Safety Stock ($SS$):** ${outputs.z_score:.4f} \\times {outputs.std_dev_protection_interval:.2f} = {outputs.safety_stock:.2f}\\text{{ units}}$")
            lines.append(f"4. **Target Order-Up-To Level ($M$):** ${outputs.expected_demand_protection_interval:.2f} + {outputs.safety_stock:.2f} = {outputs.target_inventory_level:.2f}\\text{{ units}}$")

        elif model_type == InventoryModelType.NEWSVENDOR:
            lines.append("#### Governing Formulas:")
            lines.append(r"$$C_u = P - C + g, \quad C_o = C - V, \quad CR = \frac{C_u}{C_u + C_o}$$")
            lines.append(r"$$Q^* = \mu + z^* \cdot \sigma, \quad z^* = \Phi^{-1}(CR)$$")
            lines.append("#### Step-by-Step Substitution:")
            lines.append(f"1. **Underage Cost ($C_u$):** Selling Price (${inputs_dict['selling_price']:.2f}) - Cost (${inputs_dict['purchase_cost']:.2f}) = ${outputs.cost_of_underage:.2f}/unit")
            lines.append(f"2. **Overage Cost ($C_o$):** Cost (${inputs_dict['purchase_cost']:.2f}) - Salvage (${inputs_dict.get('salvage_value', 0):.2f}) = ${outputs.cost_of_overage:.2f}/unit")
            lines.append(f"3. **Critical Ratio ($CR$):** $\\frac{{{outputs.cost_of_underage:.2f}}}{{{outputs.cost_of_underage:.2f} + {outputs.cost_of_overage:.2f}}} = {outputs.critical_ratio:.4f}$")
            lines.append(f"4. **Optimal Z-score ($z^*$):** $\\Phi^{{-1}}({outputs.critical_ratio:.4f}) = {outputs.z_score:.4f}$")
            lines.append(f"5. **Optimal Order Quantity ($Q^*$):** ${inputs_dict['demand_mean']} + ({outputs.z_score:.4f} \\times {inputs_dict['demand_std_dev']}) = {outputs.optimal_order_quantity:,.2f}\\text{{ units}}$")

        lines.append("")

        # 4. Verification Report
        lines.append("### ✅ 4. Mathematical Verification & Constraint Checks")
        for check in verification.get("checks", []):
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            lines.append(f"- **{status}**: {check['check']} — *{check['detail']}*")
        lines.append("")

        # 5. Managerial Insights
        lines.append("### 💡 5. Managerial Insights & Trade-Offs")
        if model_type == InventoryModelType.EOQ:
            lines.append("- **Holding vs Ordering Trade-Off:** At optimal $Q^*$, annual ordering cost and holding cost are virtually identical. The total cost curve is relatively flat around $Q^*$, meaning minor deviations in batch size (e.g. for palletization or packaging) will incur only negligible cost penalties.")
            lines.append("- **Square Root Rule Sensitivity:** If annual demand quadruples, the optimal order quantity only doubles, and ordering frequency doubles.")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            lines.append(f"- **Volume Leverage:** Moving to **{outputs.optimal_tier_label}** captures **${outputs.annual_cost_savings:,.2f}** in annual savings. The unit purchasing discounts substantially outweigh the increased holding cost from larger batch sizes.")
            lines.append("- **Working Capital Warning:** Ordering larger batches requires more upfront cash and physical warehouse footprint. Ensure warehouse capacity and shelf-life are adequate.")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            lines.append(f"- **Service Level Non-Linearity:** As service levels rise above 95%, safety stock requirements grow exponentially. Increasing service level from 95% ($z=1.645$) to 99% ($z=2.326$) demands a 41% surge in safety stock investment.")
            lines.append(f"- **Lead Time Reduction:** Safety stock scales with $\\sqrt{{L}}$. Shortening supplier lead time through domestic sourcing or vendor-managed inventory directly lowers safety stock buffers.")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            lines.append(f"- **Periodic vs Continuous Penalty:** Because review occurs periodically every {outputs.review_period:.0f} days, the system must buffer against uncertainty over the entire protection interval $(P + L = {outputs.protection_interval:.0f}\\text{{ days}})$, requiring **{outputs.ss_ratio_vs_continuous:.2f}x** more safety stock than continuous review.")
            lines.append("- **Operational Simplicity:** Despite higher inventory buffers, periodic review groups orders into regular supplier schedules, lowering administrative and dispatch overhead.")

        elif model_type == InventoryModelType.NEWSVENDOR:
            lines.append(f"- **Critical Ratio Dynamic:** With $CR = {outputs.critical_ratio * 100:.1f}\\%$, the high margin relative to markdown loss dictates ordering **{outputs.optimal_order_quantity - inputs_dict['demand_mean']:+.0f} units** above average demand.")
            lines.append(f"- **Value of Information (EVPI):** Perfect demand forecasting would unlock up to **${outputs.evpi:,.2f}** in added profit, setting the maximum upper bound on investments in demand forecasting systems.")

        return "\n".join(lines)

    # ==========================================================================
    # Pipeline Orchestrator: run()
    # ==========================================================================

    def run(self, question: str) -> AgentDecisionResult:
        """
        Executes the complete 6-stage architecture:
          Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
        """
        traces: List[AgentStepTrace] = []

        try:
            # 1. Identify Model
            model_type, rationale, conf = self.identify_model(question)
            traces.append(AgentStepTrace(
                step_number=1,
                step_name="Identify Model",
                summary=f"Identified {model_type.value} with confidence {conf*100:.1f}%",
                details={"model": model_type.value, "rationale": rationale, "confidence": conf}
            ))

            if model_type == InventoryModelType.UNKNOWN:
                return AgentDecisionResult(
                    question=question,
                    model_type=model_type,
                    extracted_inputs={},
                    selected_tool_name="None",
                    raw_outputs=None,
                    calculation_summary={},
                    verification_report={"verified": False, "checks": []},
                    explanation="Sorry, I could not identify an inventory decision model from your question. Please ensure your question relates to EOQ, Quantity Discounts, Safety Stock/ROP, Fixed-Period Review, or Newsvendor models.",
                    step_traces=traces,
                    success=False,
                    error_message=rationale
                )

            # 2. Extract Inputs
            extracted = self.extract_inputs(question, model_type)
            traces.append(AgentStepTrace(
                step_number=2,
                step_name="Extract Inputs",
                summary=f"Extracted {len(extracted)} input parameters from query",
                details={k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in extracted.items()}
            ))

            # 3. Select Python Tool
            tool_name, tool_func, module_path = self.select_tool(model_type)
            traces.append(AgentStepTrace(
                step_number=3,
                step_name="Select Python Tool",
                summary=f"Selected tool '{tool_name}' from {module_path}",
                details={"tool_name": tool_name, "module": module_path}
            ))

            # 4. Calculate
            outputs = self.calculate(model_type, extracted)
            traces.append(AgentStepTrace(
                step_number=4,
                step_name="Calculate",
                summary="Completed deterministic Python numerical calculation",
                details={"outputs_repr": str(outputs)}
            ))

            # 5. Verify
            verification = self.verify(model_type, extracted, outputs)
            traces.append(AgentStepTrace(
                step_number=5,
                step_name="Verify",
                summary=f"Verification status: {'PASSED' if verification['verified'] else 'FAILED'}",
                details=verification
            ))

            # 6. Explain
            explanation = self.explain(question, model_type, extracted, outputs, verification)
            traces.append(AgentStepTrace(
                step_number=6,
                step_name="Explain",
                summary="Generated executive & managerial decision explanation",
                details={"explanation_length": len(explanation)}
            ))

            # Summary dict for quick access
            calc_summary = {}
            if model_type == InventoryModelType.EOQ:
                calc_summary = {
                    "Q_star": outputs.optimal_order_quantity,
                    "orders_per_year": outputs.annual_orders_count,
                    "cycle_days": outputs.order_cycle_days,
                    "total_relevant_cost": outputs.total_relevant_cost,
                    "reorder_point": outputs.reorder_point
                }
            elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
                calc_summary = {
                    "optimal_tier": outputs.optimal_tier_label,
                    "optimal_order_quantity": outputs.optimal_order_quantity,
                    "optimal_unit_price": outputs.optimal_unit_price,
                    "min_total_annual_cost": outputs.min_total_annual_cost,
                    "annual_cost_savings": outputs.annual_cost_savings
                }
            elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
                calc_summary = {
                    "safety_stock": outputs.safety_stock,
                    "reorder_point": outputs.reorder_point,
                    "service_level": outputs.service_level,
                    "z_score": outputs.z_score
                }
            elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                calc_summary = {
                    "target_inventory_level": outputs.target_inventory_level,
                    "safety_stock": outputs.safety_stock,
                    "protection_interval": outputs.protection_interval,
                    "recommended_order_quantity": outputs.recommended_order_quantity
                }
            elif model_type == InventoryModelType.NEWSVENDOR:
                calc_summary = {
                    "optimal_order_quantity": outputs.optimal_order_quantity,
                    "critical_ratio": outputs.critical_ratio,
                    "expected_profit": outputs.expected_profit,
                    "evpi": outputs.evpi
                }

            return AgentDecisionResult(
                question=question,
                model_type=model_type,
                extracted_inputs=extracted,
                selected_tool_name=tool_name,
                raw_outputs=outputs,
                calculation_summary=calc_summary,
                verification_report=verification,
                explanation=explanation,
                step_traces=traces,
                success=True
            )

        except Exception as e:
            traces.append(AgentStepTrace(
                step_number=len(traces) + 1,
                step_name="Pipeline Exception",
                summary=f"Error encountered: {str(e)}",
                details={"error": str(e)}
            ))
            return AgentDecisionResult(
                question=question,
                model_type=model_type if 'model_type' in locals() else InventoryModelType.UNKNOWN,
                extracted_inputs=extracted if 'extracted' in locals() else {},
                selected_tool_name=tool_name if 'tool_name' in locals() else "None",
                raw_outputs=None,
                calculation_summary={},
                verification_report={"verified": False, "checks": []},
                explanation=f"### ⚠️ Calculation Error\nAn error occurred while processing the inventory decision request:\n> **{str(e)}**\n\nPlease check that your question provides the required parameters for this model.",
                step_traces=traces,
                success=False,
                error_message=str(e)
            )
