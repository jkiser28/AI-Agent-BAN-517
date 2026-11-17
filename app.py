"""
Supply Chain Analytics Inventory Decision Agent Application
============================================================
Interactive application providing natural language decision support for supply chain inventory managers.

Architecture:
Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain

Supports 5 Core Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model (Price Break)
3. Safety Stock & Reorder Point (SS & ROP)
4. Fixed-Period Review Model (Periodic Review / Order-Up-To Level)
5. Newsvendor Model (Single-Period Stochastic Demand)

Usage:
    - Interactive CLI:       python3 app.py
    - Single Question:       python3 app.py --question "We have annual demand of 12,000 units..."
    - Run All Demos:         python3 app.py --demo
    - Streamlit Web App:     streamlit run app.py
"""

import argparse
import os
import sys
from typing import Dict, Any, List

from inventory_agent import InventoryAgent, InventoryModelType

# Benchmark demonstration questions covering all 5 models
DEMO_SCENARIOS = [
    {
        "model": "Economic Order Quantity (EOQ)",
        "question": (
            "We sell 12,000 units of an industrial valve annually. "
            "Our cost to place an order is $100, and our annual holding cost is $5.00 per unit. "
            "Our operating year has 250 days, and vendor lead time is 6 days. "
            "What is our optimal order size and reorder point?"
        ),
        "notes": "Classic deterministic continuous-review lot sizing."
    },
    {
        "model": "Quantity Discount Model",
        "question": (
            "Our annual demand is 10,000 components and ordering cost is $80. "
            "Our annual inventory holding rate is 20% of purchase price. "
            "The supplier offers this price break schedule: "
            "0-499 units: $25.00, 500-999 units: $24.00, 1000+ units: $22.50. "
            "What quantity should we order to minimize total annual inventory costs?"
        ),
        "notes": "All-units quantity discount with price-break trade-off."
    },
    {
        "model": "Safety Stock & Reorder Point (SS & ROP)",
        "question": (
            "Daily demand for our flagship smartphone charger averages 120 units with a standard deviation of 25 units. "
            "Supplier replenishment lead time averages 9 days. "
            "Management mandates a 95% cycle service level to protect against stockouts. "
            "What should our safety stock and reorder point be?"
        ),
        "notes": "Stochastic demand with continuous review (s, Q) policy."
    },
    {
        "model": "Fixed-Period Review Model (Order-Up-To)",
        "question": (
            "We review inventory for store fixtures every 14 days. "
            "The supplier takes 7 days to deliver after receiving an order. "
            "Daily demand averages 30 units with a standard deviation of 8 units. "
            "We target a 98% cycle service level. "
            "Currently, we have 180 units on hand and 100 units on order with 0 backorders. "
            "What is our target order-up-to level S, and how many units should we order today?"
        ),
        "notes": "Periodic review (T, S) system protecting over (T + L)."
    },
    {
        "model": "Newsvendor Model",
        "question": (
            "We are purchasing commemorative holiday gift baskets for a single seasonal event. "
            "Each basket sells for $90 at retail and costs us $40 to procure. "
            "Unsold baskets after the holiday can be liquidated to a discounter for a salvage value of $15. "
            "Event demand is normally distributed with a mean of 1,500 baskets and a standard deviation of 300 baskets. "
            "How many gift baskets should we purchase to maximize expected profit?"
        ),
        "notes": "Perishable single-period critical fractile optimization."
    }
]


def print_banner():
    banner = """
================================================================================
   SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT
   Architecture: Question -> Identify -> Extract -> Tool -> Calculate -> Verify -> Explain
================================================================================
    """
    print(banner)


def display_agent_response(response, verbose: bool = True):
    """Format and print agent response to terminal."""
    print("\n" + "=" * 80)
    print(f"🎯 1. MODEL IDENTIFIED: {response.model_type.value}")
    print(f"   Confidence Score: {response.confidence:.1%}")
    print(f"   Selected Tool:    {response.tool_name}")
    print("=" * 80)

    if not response.success:
        print(f"\n❌ ERROR: {response.error_message}")
        print(f"\nExplanation / Missing Details:\n{response.detailed_explanation}")
        return

    print(f"\n📢 EXECUTIVE DECISION:")
    print(f"   {response.executive_summary}\n")

    if verbose:
        print("-" * 80)
        print("🔍 2. EXTRACTED INPUT PARAMETERS:")
        for k, v in response.cleaned_inputs.items():
            if k == "discount_schedule":
                print(f"   - {k}: {len(v)} pricing tiers parsed")
                for t in v:
                    max_s = f"{t['max_qty']:,.0f}" if t.get('max_qty') is not None else "∞"
                    print(f"       * Tier range [{t['min_qty']:,.0f} - {max_s}]: ${t['price']:,.2f}/unit")
            elif v is not None:
                print(f"   - {k}: {v}")

        print("\n🧮 3. NUMERICAL CALCULATION RESULTS & STEP-BY-STEP DERIVATION:")
        step_by_step = response.calculation_results.get("step_by_step_calculation")
        if step_by_step and isinstance(step_by_step, list):
            for step in step_by_step:
                print(f"   {step}")
        else:
            excluded_keys = {
                "evaluated_tiers", "inputs_extracted", "inputs", "variable_definitions",
                "formula", "formulas", "formula_with_substituted_values",
                "intermediate_calculations", "final_answer", "verification_checks", "assumptions", "tool",
                "step_by_step_calculation", "step_by_step_derivation"
            }
            for k, v in response.calculation_results.items():
                if k not in excluded_keys and not isinstance(v, (dict, list)):
                    print(f"   - {k}: {v}")

        print("\n✅ 4. VERIFICATION AUDIT & SANITY CHECKS:")
        for v in response.verifications:
            icon = "✅" if v.status == "PASS" else ("⚠️" if v.status == "WARN" else ("❌" if v.status == "FAIL" else "ℹ️"))
            print(f"   [{icon} {v.status}] {v.name}: {v.detail}")

        print("\n" + "-" * 80)
        print("📄 5. DETAILED ANALYTICAL REPORT:")
        print(response.detailed_explanation)

    print("=" * 80 + "\n")


def run_demo(agent: InventoryAgent):
    """Execute all pre-configured benchmark scenarios."""
    print("\n🚀 Running Supply Chain Inventory Agent Benchmark Scenarios...")
    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'#' * 80}")
        print(f"SCENARIO {i}/5: {scenario['model'].upper()}")
        print(f"Notes: {scenario['notes']}")
        print(f"Input Question: \"{scenario['question']}\"")
        print(f"{'#' * 80}")

        response = agent.process_query(scenario["question"])
        display_agent_response(response, verbose=True)


def run_interactive(agent: InventoryAgent):
    """Run interactive question-answer REPL."""
    print_banner()
    print("Type your supply chain inventory question below, or use:")
    print("  'demo'      -> Run all 5 benchmark scenarios")
    print("  'examples'  -> Show example questions")
    print("  'help'      -> Show system capabilities")
    print("  'quit'      -> Exit application\n")

    while True:
        try:
            user_input = input("Inventory-Agent> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["quit", "exit", "q"]:
            print("Exiting Inventory Decision Agent. Goodbye!")
            break
        elif cmd in ["demo", "run demo"]:
            run_demo(agent)
            continue
        elif cmd in ["examples", "example"]:
            print("\n📋 Sample Questions You Can Ask:")
            for idx, sc in enumerate(DEMO_SCENARIOS, 1):
                print(f"\n{idx}. [{sc['model']}]:\n   \"{sc['question']}\"")
            print()
            continue
        elif cmd in ["help", "models"]:
            print("\n🛠️ Supported Inventory Models:")
            print("  1. Economic Order Quantity (EOQ): Balance setup/ordering vs holding costs")
            print("  2. Quantity Discount Model: Multi-tier price-break evaluation")
            print("  3. Safety Stock & Reorder Point (ROP): Lead time & demand stochasticity (s, Q)")
            print("  4. Fixed-Period Review Model: Periodic review (T, S) order-up-to target")
            print("  5. Newsvendor Model: Single-period stochastic critical fractile optimization\n")
            continue

        # Process user question through the 6-stage pipeline
        response = agent.process_query(user_input)
        display_agent_response(response, verbose=True)


def analyze_inventory_dataframe(df: Any) -> Dict[str, Any]:
    """
    Validate and analyze an inventory pandas DataFrame.
    Required columns: Product, Category, Units, Unit Cost.

    Calculations performed via pandas:
    1. Total Units
    2. Average Unit Cost
    3. Product with highest Units
    4. Group data by Category and calculate total Units for each Category
    5. New calculated column: Inventory Value = Units x Unit Cost
    6. Total Inventory Value across all products
    """
    required_cols = ["Product", "Category", "Units", "Unit Cost"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "is_valid": False,
            "missing_columns": missing_cols,
            "required_columns": required_cols,
            "error_message": f"Missing required column(s): {', '.join(missing_cols)}. Expected: {', '.join(required_cols)}",
        }

    try:
        import pandas as pd
    except ImportError:
        return {
            "is_valid": False,
            "missing_columns": [],
            "required_columns": required_cols,
            "error_message": "pandas library is required for data calculations.",
        }

    # Work on a copy of the uploaded dataframe
    df_result = df.copy()

    # Safely convert Units and Unit Cost to numeric using pandas
    df_result["Units"] = pd.to_numeric(df_result["Units"], errors="coerce").fillna(0.0)
    df_result["Unit Cost"] = pd.to_numeric(df_result["Unit Cost"], errors="coerce").fillna(0.0)

    # 1. Calculate total Units
    total_units = float(df_result["Units"].sum())

    # 2. Calculate average Unit Cost
    avg_unit_cost = float(df_result["Unit Cost"].mean()) if len(df_result) > 0 else 0.0

    # 3. Identify the Product with the highest Units
    if len(df_result) > 0:
        max_idx = df_result["Units"].idxmax()
        highest_units_product = str(df_result.loc[max_idx, "Product"])
        highest_units_count = float(df_result.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group data by Category and calculate total Units for each Category
    category_summary = (
        df_result.groupby("Category")["Units"]
        .sum()
        .reset_index()
        .rename(columns={"Units": "Total Units"})
    )

    # 5. Create new calculated column: Inventory Value = Units x Unit Cost
    df_result["Inventory Value"] = df_result["Units"] * df_result["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(df_result["Inventory Value"].sum())

    return {
        "is_valid": True,
        "dataframe": df_result,
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "column_names": list(df.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "product_highest_units": highest_units_product,
        "highest_units_count": highest_units_count,
        "category_summary": category_summary,
        "total_inventory_value": total_inventory_value,
    }


def classify_inventory_dataframe(df: Any) -> Dict[str, Any]:
    """
    Validate and classify inventory items using multicriteria classification (Sales ($) and Hits).
    Supports A, B, C, D classifications for both Sales ($) and Hits.

    Column Titles for Results:
    - Cumulative Sales ($) Total
    - Percentage Sales ($)
    - Sales ($) Classification
    - Cumulative Hits Total
    - Percentage Hits
    - Hits Classification
    - Multicriteria Class

    Classification Criteria:
    - A items: 0% < x <= 60%
    - B items: 60% < x <= 80%
    - C items: 80% < x <= 90%
    - D items: 90% < x

    Calculations performed via pandas:
    1. Identifies and validates Sales ($) and Hits columns flexibly.
       - Detects explicit 'Sales ($)', 'Sales', 'Annual Sales', 'Revenue ($)'.
       - If missing but 'Units' and 'Unit Cost' exist, computes Sales ($) = Units * Unit Cost.
       - Detects explicit 'Hits', 'Hit', 'Picks', 'Frequency', 'Transactions', 'Orders'.
       - If missing but 'Units' exists, uses 'Units' as Hits proxy.
       - Detects Item identifier: 'Item Name', 'Product', 'Item Number', 'Item', 'Description', 'SKU', or first column.
    2. Cleans currency formatting, dollar signs, commas, and whitespace.
    3. Sorts items from largest to smallest in terms of Sales ($).
    4. Cumulative Sales ($) Total: starting with largest Sales ($) equaling itself,
       adding each next largest Sales ($) down the list.
    5. Percentage Sales ($): (Cumulative Sales ($) Total / Total Sales) * 100.
    6. Classifies Sales ($) items:
       - A: 0% < x <= 60%
       - B: 60% < x <= 80%
       - C: 80% < x <= 90%
       - D: 90% < x
    7. Sorts items from largest to smallest in terms of Hits before calculating:
       Cumulative Hits Total: running sum of Hits.
       Percentage Hits: (Cumulative Hits Total / Total Hits) * 100.
       Classifies Hits items:
       - A: 0% < x <= 60%
       - B: 60% < x <= 80%
       - C: 80% < x <= 90%
       - D: 90% < x
    8. Forms Multicriteria Class (e.g. A-A, A-B, B-C, D-D).
    9. Generates 4x4 cross-tabulation matrix (pd.crosstab) for multicriteria inventory control.
    10. Generates Class summary distribution tables for both Sales ($) and Hits.
    """
    # Identify columns flexibly
    sales_col = None
    for c in getattr(df, "columns", []):
        c_clean = str(c).strip().lower()
        if c_clean in ["sales ($)", "sales", "sales($)", "sales (usd)", "annual sales", "sales_$", "revenue ($)", "revenue"]:
            sales_col = c
            break
        elif "sale" in c_clean or "revenue" in c_clean:
            sales_col = c
            break

    hits_col = None
    for c in getattr(df, "columns", []):
        c_clean = str(c).strip().lower()
        if c_clean in ["hits", "hit", "picks", "pick hits", "order frequency", "transactions", "frequency", "orders"]:
            hits_col = c
            break
        elif "hit" in c_clean or "pick" in c_clean:
            hits_col = c
            break

    # Fallback: if Units and Unit Cost exist, Sales ($) can be derived
    if not sales_col:
        has_units = any("unit" in str(c).lower() and "cost" not in str(c).lower() and "price" not in str(c).lower() for c in getattr(df, "columns", []))
        has_cost = any("cost" in str(c).lower() or "price" in str(c).lower() for c in getattr(df, "columns", []))
        if has_units and has_cost:
            sales_col = "Sales ($)"

    # Fallback: if Units exists, Hits can be derived from Units
    if not hits_col:
        has_units = any("unit" in str(c).lower() and "cost" not in str(c).lower() and "price" not in str(c).lower() for c in getattr(df, "columns", []))
        if has_units:
            hits_col = "Hits"

    # Validate missing columns before attempting dataframe operations
    missing_cols = []
    if not sales_col:
        missing_cols.append("Sales ($)")
    if not hits_col:
        missing_cols.append("Hits")

    if missing_cols:
        return {
            "is_valid": False,
            "missing_columns": missing_cols,
            "required_columns": ["Item Name", "Sales ($)", "Hits"],
            "error_message": f"Missing required column(s) for classification: {', '.join(missing_cols)}. Expected columns: Item Number, Item Name, UoM, Sales ($), Hits.",
        }

    try:
        import pandas as pd
    except ImportError:
        return {
            "is_valid": False,
            "missing_columns": [],
            "required_columns": ["Item Name", "Sales ($)", "Hits"],
            "error_message": "pandas library is required for data calculations.",
        }

    if not hasattr(df, "copy"):
        return {
            "is_valid": False,
            "missing_columns": [],
            "required_columns": ["Item Name", "Sales ($)", "Hits"],
            "error_message": "Input must be a valid pandas DataFrame.",
        }

    df_result = df.copy()

    # Identify item identifier column
    name_col = None
    for c in df_result.columns:
        c_clean = str(c).strip().lower()
        if c_clean in ["item name", "item_name", "product", "product name", "item", "description", "item description"]:
            name_col = c
            break
    if not name_col:
        for c in df_result.columns:
            c_clean = str(c).strip().lower()
            if any(k in c_clean for k in ["number", "num", "sku", "id", "part", "code"]):
                name_col = c
                break
    if not name_col and len(df_result.columns) > 0:
        name_col = df_result.columns[0]

    # Compute derived Sales ($) if needed
    if sales_col == "Sales ($)" and "Sales ($)" not in df_result.columns:
        units_c = next((c for c in df_result.columns if "unit" in str(c).lower() and "cost" not in str(c).lower() and "price" not in str(c).lower()), None)
        cost_c = next((c for c in df_result.columns if "cost" in str(c).lower() or "price" in str(c).lower()), None)
        if units_c and cost_c:
            def _clean_n(s):
                if s.dtype == object:
                    clean = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
                    return pd.to_numeric(clean, errors="coerce").fillna(0.0)
                return pd.to_numeric(s, errors="coerce").fillna(0.0)
            df_result["Sales ($)"] = _clean_n(df_result[units_c]) * _clean_n(df_result[cost_c])

    # Compute derived Hits if needed
    if hits_col == "Hits" and "Hits" not in df_result.columns:
        units_c = next((c for c in df_result.columns if "unit" in str(c).lower() and "cost" not in str(c).lower() and "price" not in str(c).lower()), None)
        if units_c:
            df_result["Hits"] = df_result[units_c]

    # Clean numeric values
    def clean_series(s):
        if s.dtype == object:
            clean = s.astype(str).str.replace("$", "", regex=False)
            clean = clean.str.replace(",", "", regex=False).str.strip()
            return pd.to_numeric(clean, errors="coerce").fillna(0.0)
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    df_result["_sales_clean"] = clean_series(df_result[sales_col])
    df_result["_hits_clean"] = clean_series(df_result[hits_col])

    # 1. Sort items from largest to smallest in terms of Sales ($)
    df_sorted = df_result.sort_values(
        by=["_sales_clean", "_hits_clean"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # 2. Cumulative Sales ($) Total: starting with largest Sales ($) equaling itself
    df_sorted["Cumulative Sales ($) Total"] = df_sorted["_sales_clean"].cumsum()

    # 3. Percentage Sales ($): cumulative value divided by total sales value
    total_sales = float(df_sorted["_sales_clean"].sum())
    if total_sales > 0:
        df_sorted["Percentage Sales ($)"] = (df_sorted["Cumulative Sales ($) Total"] / total_sales) * 100.0
    else:
        df_sorted["Percentage Sales ($)"] = 0.0

    # Aliases for backward compatibility
    df_sorted["Cumulative Total"] = df_sorted["Cumulative Sales ($) Total"]
    df_sorted["Percentage"] = df_sorted["Percentage Sales ($)"]

    # 4. Classify based on criteria:
    # - A items 0% < x <= 60%
    # - B items 60% < x <= 80%
    # - C items 80% < x <= 90%
    # - D items 90% < x
    def assign_class(pct: float) -> str:
        if pct <= 60.0:
            return "A"
        elif pct <= 80.0:
            return "B"
        elif pct <= 90.0:
            return "C"
        else:
            return "D"

    df_sorted["Sales ($) Classification"] = df_sorted["Percentage Sales ($)"].apply(assign_class)
    df_sorted["Classification"] = df_sorted["Sales ($) Classification"]

    # 5. Hits classification: Sort Hits values from largest to smallest before calculating
    total_hits = float(df_sorted["_hits_clean"].sum())
    df_hits_sorted = df_sorted.sort_values(by=["_hits_clean", "_sales_clean"], ascending=[False, False]).copy()
    df_hits_sorted["Cumulative Hits Total"] = df_hits_sorted["_hits_clean"].cumsum()
    if total_hits > 0:
        df_hits_sorted["Percentage Hits"] = (df_hits_sorted["Cumulative Hits Total"] / total_hits) * 100.0
    else:
        df_hits_sorted["Percentage Hits"] = 0.0
    df_hits_sorted["Hits Classification"] = df_hits_sorted["Percentage Hits"].apply(assign_class)

    # Map Hits metrics back to df_sorted order
    df_sorted["Cumulative Hits Total"] = df_hits_sorted["Cumulative Hits Total"].loc[df_sorted.index]
    df_sorted["Percentage Hits"] = df_hits_sorted["Percentage Hits"].loc[df_sorted.index]
    df_sorted["Hits Classification"] = df_hits_sorted["Hits Classification"].loc[df_sorted.index]

    # Aliases for hits
    df_sorted["Cumulative Hits"] = df_sorted["Cumulative Hits Total"]
    df_sorted["Hits Percentage"] = df_sorted["Percentage Hits"]

    # 6. Multicriteria composite classification
    df_sorted["Multicriteria Class"] = df_sorted["Sales ($) Classification"] + "-" + df_sorted["Hits Classification"]

    # 7. Sales ($) Class summary distribution table
    classes = ["A", "B", "C", "D"]
    total_items = len(df_sorted)
    sales_summary_rows = []
    sales_policy_map = {
        "A": "Strict inventory control, continuous review, tight replenishment, high target service level",
        "B": "Moderate control, periodic review, standard safety stock buffer, automated reordering",
        "C": "Simplified control, larger order quantities, visual review / periodic batching",
        "D": "Candidate for rationalization/liquidation, make-to-order, minimal buffer stock"
    }
    criteria_map = {
        "A": "0% < x ≤ 60%",
        "B": "60% < x ≤ 80%",
        "C": "80% < x ≤ 90%",
        "D": "90% < x"
    }

    cum_sales_pct_running = 0.0
    for cls in classes:
        subset = df_sorted[df_sorted["Sales ($) Classification"] == cls]
        cnt = len(subset)
        pct_items = (cnt / total_items * 100.0) if total_items > 0 else 0.0
        cls_sales = float(subset["_sales_clean"].sum())
        pct_sales = (cls_sales / total_sales * 100.0) if total_sales > 0 else 0.0
        cum_sales_pct_running += pct_sales

        sales_summary_rows.append({
            "Class": cls,
            "Sales Criteria": criteria_map[cls],
            "Item Count": cnt,
            "% of Items": round(pct_items, 1),
            "Total Sales ($)": cls_sales,
            "% of Sales": round(pct_sales, 1),
            "Cumulative Sales %": round(cum_sales_pct_running, 1),
            "Management Policy": sales_policy_map[cls]
        })
    sales_summary_df = pd.DataFrame(sales_summary_rows)

    # 8. Hits Class summary distribution table
    hits_summary_rows = []
    hits_policy_map = {
        "A": "Fast Movers: Prime warehouse picking slotting (golden zone), frequent replenishment",
        "B": "Medium Movers: Standard shelf/rack slotting, periodic picking replenishment",
        "C": "Slow Movers: Upper or lower rack storage, batch picking consolidation",
        "D": "Very Slow Movers / Dead Stock: Reserve or remote storage, review for obsolete stock purge"
    }

    cum_hits_pct_running = 0.0
    for cls in classes:
        subset = df_sorted[df_sorted["Hits Classification"] == cls]
        cnt = len(subset)
        pct_items = (cnt / total_items * 100.0) if total_items > 0 else 0.0
        cls_hits = float(subset["_hits_clean"].sum())
        pct_hits = (cls_hits / total_hits * 100.0) if total_hits > 0 else 0.0
        cum_hits_pct_running += pct_hits

        hits_summary_rows.append({
            "Class": cls,
            "Hits Criteria": criteria_map[cls],
            "Item Count": cnt,
            "% of Items": round(pct_items, 1),
            "Total Hits": cls_hits,
            "% of Hits": round(pct_hits, 1),
            "Cumulative Hits %": round(cum_hits_pct_running, 1),
            "Warehouse Policy": hits_policy_map[cls]
        })
    hits_summary_df = pd.DataFrame(hits_summary_rows)

    # 9. 4x4 Multicriteria Cross-Tabulation Matrix
    base_matrix = pd.crosstab(
        df_sorted["Sales ($) Classification"],
        df_sorted["Hits Classification"]
    )
    for cls in ["A", "B", "C", "D"]:
        if cls not in base_matrix.index:
            base_matrix.loc[cls] = 0
        if cls not in base_matrix.columns:
            base_matrix[cls] = 0
    crosstab_matrix = base_matrix.reindex(index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"]).fillna(0).astype(int)

    # Matrix with row/col totals
    crosstab_with_totals = crosstab_matrix.copy()
    crosstab_with_totals["Total"] = crosstab_with_totals.sum(axis=1)
    totals_row = crosstab_with_totals.sum(axis=0)
    totals_row.name = "Total"
    crosstab_with_totals = pd.concat([crosstab_with_totals, pd.DataFrame([totals_row])])

    # Clean internal temporary columns for display
    df_display = df_sorted.drop(columns=["_sales_clean", "_hits_clean"])

    return {
        "is_valid": True,
        "dataframe": df_display,
        "raw_sorted_df": df_sorted,
        "num_rows": int(df.shape[0]),
        "num_cols": int(df_display.shape[1]),
        "column_names": list(df_display.columns),
        "total_sales": total_sales,
        "total_hits": total_hits,
        "total_items": total_items,
        "class_summary": sales_summary_df,
        "sales_summary": sales_summary_df,
        "hits_summary": hits_summary_df,
        "crosstab_matrix": crosstab_matrix,
        "crosstab_with_totals": crosstab_with_totals,
        "sales_col": sales_col,
        "hits_col": hits_col,
        "name_col": name_col,
    }


def try_run_streamlit(agent: InventoryAgent):
    """Render a Streamlit web application if run via 'streamlit run app.py'."""
    try:
        import streamlit as st
    except ImportError:
        return False

    # Check if streamlit is active in the call stack
    try:
        import streamlit.runtime.scriptrunner as scriptrunner
        is_streamlit = scriptrunner.get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if not is_streamlit:
        return False

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide"
    )

    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "**Architecture:** *Customer Question → Identify Model → Extract Inputs → "
        "Select Python Tool → Calculate → Verify → Explain*"
    )

    # Sidebar: Model Selector and Quick Examples
    st.sidebar.header("🎯 Preset Scenarios")
    scenario_choice = st.sidebar.selectbox(
        "Load an Example Benchmark:",
        ["Custom Question"] + [f"{i}. {s['model']}" for i, s in enumerate(DEMO_SCENARIOS, 1)]
    )

    # DATA UPLOAD: Section called "Data Upload" as the last section in the sidebar, near bottom
    st.sidebar.markdown("---")
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload an inventory CSV file for classification (Item Name, Sales ($), Hits) or product valuation (Product, Category, Units, Unit Cost)"
    )

    default_text = ""
    if scenario_choice != "Custom Question":
        idx = int(scenario_choice.split(".")[0]) - 1
        default_text = DEMO_SCENARIOS[idx]["question"]

    user_query = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text,
        height=140,
        placeholder="e.g. We have annual demand of 15,000 units. Ordering cost is $80 per order, and holding cost is $3.50 per unit/year. What is our EOQ?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🚀 Analyze & Solve", type="primary")

    if submit and user_query.strip():
        with st.spinner("Processing through 6-stage analytical pipeline..."):
            resp = agent.process_query(user_query)

        if not resp.success:
            st.error(f"❌ Analysis Failed: {resp.error_message}")
            st.markdown(resp.detailed_explanation)
        else:
            # Display Executive Summary
            st.success(f"**Identified Model:** {resp.model_type.value} (Confidence: {resp.confidence:.0%})")
            st.info(f"💡 {resp.executive_summary}")

            # Top KPI metrics
            st.subheader("Key Inventory Decision Metrics")
            kpi_cols = st.columns(4)

            calc = resp.calculation_results
            if resp.model_type == InventoryModelType.EOQ:
                kpi_cols[0].metric("Optimal Order Size (Q*)", f"{calc['eoq']:,.0f} units")
                kpi_cols[1].metric("Orders Per Year", f"{calc['orders_per_year']:.1f}")
                kpi_cols[2].metric("Total Inventory Cost", f"${calc['total_inventory_cost']:,.2f}")
                if calc.get("reorder_point") is not None:
                    kpi_cols[3].metric("Reorder Point (ROP)", f"{calc['reorder_point']:,.0f} units")
                else:
                    kpi_cols[3].metric("Cycle Time", f"{calc['cycle_time_days']:.1f} days")

            elif resp.model_type == InventoryModelType.QUANTITY_DISCOUNT:
                kpi_cols[0].metric("Optimal Order Qty", f"{calc['optimal_order_quantity']:,.0f} units")
                kpi_cols[1].metric("Selected Tier", f"Tier {calc['optimal_tier']} (${calc['optimal_unit_price']:,.2f})")
                kpi_cols[2].metric("Total Annual Cost", f"${calc['optimal_total_cost']:,.2f}")
                kpi_cols[3].metric("Annual Savings", f"${calc['annual_savings_vs_base']:,.2f} ({calc['percentage_savings']:.1f}%)")

            elif resp.model_type == InventoryModelType.SAFETY_STOCK_ROP:
                kpi_cols[0].metric("Safety Stock (SS)", f"{calc['safety_stock']:,.0f} units")
                kpi_cols[1].metric("Reorder Point (ROP)", f"{calc['reorder_point']:,.0f} units")
                kpi_cols[2].metric("Cycle Service Level", f"{calc['service_level']:.1%}")
                kpi_cols[3].metric("Stockout Risk", f"{calc['stockout_risk']:.1%}")

            elif resp.model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                kpi_cols[0].metric("Order-Up-To Level (S)", f"{calc['order_up_to_level_S']:,.0f} units")
                kpi_cols[1].metric("Order Quantity (Q)", f"{calc['recommended_order_quantity']:,.0f} units")
                kpi_cols[2].metric("Safety Stock", f"{calc['safety_stock']:,.0f} units")
                kpi_cols[3].metric("Protection Interval", f"{calc['protection_interval']:.0f} periods")

            elif resp.model_type == InventoryModelType.NEWSVENDOR:
                kpi_cols[0].metric("Optimal Order Qty (Q*)", f"{calc['optimal_order_quantity']:,.0f} units")
                kpi_cols[1].metric("Critical Fractile", f"{calc['critical_fractile']:.1%}")
                kpi_cols[2].metric("Expected Profit", f"${calc['expected_profit']:,.2f}")
                kpi_cols[3].metric("EVPI", f"${calc['expected_value_of_perfect_information_EVPI']:,.2f}")

            # Tabs for Detailed Breakdown, Quality Verification, and Raw JSON
            tab1, tab2, tab3 = st.tabs(["📄 Step-by-Step Report", "✅ Verification Audit", "🔍 Raw Model Inputs & Output"])

            with tab1:
                st.markdown(resp.detailed_explanation)

            with tab2:
                for v in resp.verifications:
                    if v.status == "PASS":
                        st.success(f"**{v.name}**: {v.detail}")
                    elif v.status == "WARN":
                        st.warning(f"**{v.name}**: {v.detail}")
                    else:
                        st.error(f"**{v.name}**: {v.detail}")

            with tab3:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Extracted Inputs:")
                    st.json(resp.cleaned_inputs)
                with c2:
                    st.write("Calculation Output:")
                    st.json(resp.calculation_results)

    # DATA PROCESSING & ANALYSIS SECTION (When CSV is uploaded)
    if uploaded_file is not None:
        st.markdown("---")
        st.header("📊 Inventory Data Analysis & Classification")

        try:
            import pandas as pd
        except ImportError:
            st.error("❌ 'pandas' library is required to process CSV files. Please install it with `pip install pandas`.")
            return True

        try:
            # 1. Read CSV file into a pandas dataframe
            df_uploaded = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return True

        # Check schemas: Classification (Sales + Hits) vs Product Valuation (Product, Category, Units, Unit Cost)
        is_classification = any("sale" in c.lower() for c in df_uploaded.columns) and any("hit" in c.lower() for c in df_uploaded.columns)
        is_product_analysis = all(c in df_uploaded.columns for c in ["Product", "Category", "Units", "Unit Cost"])

        def render_product_analysis_view(df: Any):
            analysis = analyze_inventory_dataframe(df)
            if not analysis["is_valid"]:
                st.error(
                    f"❌ Invalid Product Analysis CSV: Missing required column(s): **{', '.join(analysis['missing_columns'])}**.\n\n"
                    f"Required columns are: **{', '.join(analysis['required_columns'])}**."
                )
                return

            st.subheader("📋 Dataset Overview")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("Number of Rows", f"{analysis['num_rows']:,}")
            meta_col2.metric("Number of Columns", f"{analysis['num_cols']:,}")
            meta_col3.metric("Column Count", f"{len(analysis['column_names']):,}")
            st.caption(f"**Column Names:** {', '.join(analysis['column_names'])}")

            with st.expander("🔍 View Raw Uploaded DataFrame", expanded=False):
                st.dataframe(df, use_container_width=True)

            st.subheader("📈 Inventory Analysis Results")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Total Units", f"{analysis['total_units']:,.0f}")
            kpi_col2.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
            kpi_col3.metric(
                "Product with Highest Units",
                analysis["product_highest_units"],
                help=f"{analysis['highest_units_count']:,.0f} units"
            )
            kpi_col4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

            cat_col, val_col = st.columns([1, 2])
            with cat_col:
                st.markdown("#### 🏷️ Total Units by Category")
                st.dataframe(
                    analysis["category_summary"].style.format({"Total Units": "{:,.0f}"}),
                    use_container_width=True,
                    hide_index=True
                )

            with val_col:
                st.markdown("#### 📦 Product Inventory Value Details")
                st.caption("Includes calculated column: `Inventory Value = Units × Unit Cost`")

            df_enhanced = analysis["dataframe"]
            formatted_df = df_enhanced.copy()
            formatted_df["Unit Cost"] = formatted_df["Unit Cost"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            formatted_df["Inventory Value"] = formatted_df["Inventory Value"].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
            formatted_df["Units"] = formatted_df["Units"].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
            st.dataframe(formatted_df, use_container_width=True)

        def render_classification_view(df: Any):
            res = classify_inventory_dataframe(df)
            if not res["is_valid"]:
                st.error(f"❌ Classification Error: {res['error_message']}")
                st.info(f"Columns found in uploaded CSV: `{', '.join(list(df.columns))}`")
                return

            st.subheader("📋 Multicriteria Classification Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total SKUs / Items", f"{res['total_items']:,}")
            c2.metric("Total Annual Sales", f"${res['total_sales']:,.2f}")
            c3.metric("Total Pick Hits", f"{res['total_hits']:,.0f}")
            class_a_sales = res["class_summary"].loc[res["class_summary"]["Class"] == "A", "Total Sales ($)"].values[0] if len(res["class_summary"]) > 0 else 0.0
            class_a_pct = (class_a_sales / res["total_sales"] * 100) if res["total_sales"] > 0 else 0.0
            c4.metric("Class A Revenue Share", f"{class_a_pct:.1f}%", f"${class_a_sales:,.2f}")

            with st.expander("🔍 View Raw Uploaded DataFrame", expanded=False):
                st.dataframe(df, use_container_width=True)

            # 4-Class summary distribution table
            st.markdown("### 🏆 Inventory Classification Summary (A, B, C, D)")
            st.markdown(
                "**Classification Criteria:** "
                "• **Class A**: 0% < x ≤ 60% | "
                "• **Class B**: 60% < x ≤ 80% | "
                "• **Class C**: 80% < x ≤ 90% | "
                "• **Class D**: 90% < x"
            )

            summary_display = res["class_summary"].copy()
            summary_display["Total Sales ($)"] = summary_display["Total Sales ($)"].apply(lambda x: f"${x:,.2f}")
            summary_display["% of Items"] = summary_display["% of Items"].apply(lambda x: f"{x:.1f}%")
            summary_display["% of Sales"] = summary_display["% of Sales"].apply(lambda x: f"{x:.1f}%")
            summary_display["Total Hits"] = summary_display["Total Hits"].apply(lambda x: f"{x:,.0f}")
            summary_display["% of Hits"] = summary_display["% of Hits"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

            # Multicriteria 2D Matrix (Sales Class vs Hits Class)
            st.markdown("### 🔲 Multicriteria Matrix: Sales ($) vs. Hits (Velocity)")
            m_col1, m_col2 = st.columns([1, 1])
            with m_col1:
                st.markdown("**SKU Count by Class Combination (Sales Class Rows × Hits Class Columns)**")
                st.dataframe(res["crosstab_with_totals"], use_container_width=True)

            with m_col2:
                with st.expander("📖 Managerial Policy Guide for Matrix Quadrants", expanded=True):
                    st.markdown(
                        "- **A-A (High Sales, High Hits):** Fast-moving revenue drivers. Strict daily monitoring, continuous replenishment, tight supplier integration.\n"
                        "- **A-D (High Sales, Low Hits):** High dollar, slow moving / lumpy demand. Strict managerial review, make-to-order, minimal safety stock.\n"
                        "- **D-A (Low Sales, High Hits):** Cheap fast movers (fasteners, supplies). Bulk ordering, two-bin visual Kanban, high service levels.\n"
                        "- **D-D (Low Sales, Low Hits):** Dead stock or slow moving. Review for SKU rationalization, liquidation, or zero-inventory on-demand sourcing."
                    )

            # Full Classified Table
            st.markdown("### 📑 Full Classified Inventory Table")
            st.caption("Sorted descending by Sales ($). Displays Cumulative Total, Cumulative Percentage, Sales Classification, Hits Classification, and Multicriteria Class.")

            df_out = res["dataframe"].copy()
            sales_c = res["sales_col"]
            hits_c = res["hits_col"]

            df_out_display = df_out.copy()
            df_out_display[sales_c] = df_out_display[sales_c].apply(
                lambda x: f"${float(str(x).replace('$', '').replace(',', '')):,.2f}" if pd.notnull(x) else ""
            )
            df_out_display["Cumulative Total"] = df_out_display["Cumulative Total"].apply(
                lambda x: f"${x:,.2f}" if pd.notnull(x) else ""
            )
            df_out_display["Percentage"] = df_out_display["Percentage"].apply(
                lambda x: f"{x:.1f}%" if pd.notnull(x) else ""
            )
            df_out_display[hits_c] = df_out_display[hits_c].apply(
                lambda x: f"{float(str(x).replace(',', '')):,.0f}" if pd.notnull(x) else ""
            )

            st.dataframe(df_out_display, use_container_width=True)

            # Download classified dataset
            csv_data = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Inventory CSV",
                data=csv_data,
                file_name="classified_inventory.csv",
                mime="text/csv",
                help="Export the sorted inventory with Cumulative Total, Percentage, and Classifications"
            )

        if is_classification and is_product_analysis:
            tab_class, tab_prod = st.tabs(["🏷️ Multicriteria Classification (A/B/C/D)", "📦 Product Value Analysis"])
            with tab_class:
                render_classification_view(df_uploaded)
            with tab_prod:
                render_product_analysis_view(df_uploaded)
        elif is_classification:
            render_classification_view(df_uploaded)
        elif is_product_analysis:
            render_product_analysis_view(df_uploaded)
        else:
            st.error(
                "❌ Invalid CSV File: The uploaded dataset does not match either supported schema:\n\n"
                "1. **Inventory Classification:** Requires `Item Name`, `Sales ($)`, and `Hits` (optional: `Item Number`, `UoM`).\n"
                "2. **Product Valuation:** Requires `Product`, `Category`, `Units`, and `Unit Cost`."
            )
            st.info(f"Columns found in uploaded CSV: `{', '.join(list(df_uploaded.columns))}`")

    return True


def main():
    agent = InventoryAgent()

    # Check if executed under Streamlit runner
    if try_run_streamlit(agent):
        return

    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("--question", "-q", type=str, help="Single question to evaluate")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demonstration scenarios")
    parser.add_argument("--brief", action="store_true", help="Print brief summary output only")
    args = parser.parse_args()

    if args.demo:
        print_banner()
        run_demo(agent)
    elif args.question:
        resp = agent.process_query(args.question)
        display_agent_response(resp, verbose=not args.brief)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
