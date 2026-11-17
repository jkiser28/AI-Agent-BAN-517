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
