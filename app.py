"""
Supply Chain Analytics Inventory Decision Agent Application
===========================================================
Dual-mode application:
  1. Interactive CLI Console: python3 app.py [--question "..."] [--demo]
  2. Streamlit Web Dashboard: streamlit run app.py

Implements the 6-stage architecture:
  Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

import sys
import argparse
from typing import Optional, List, Dict, Any

from inventory_agent import InventoryDecisionAgent, InventoryModelType, AgentDecisionResult
from inventory_tools import (
    calculate_eoq, EOQInputs,
    calculate_quantity_discount, QuantityDiscountInputs, DiscountTier,
    calculate_safety_stock_and_rop, SafetyStockROPInputs,
    calculate_fixed_period_review, FixedPeriodReviewInputs,
    calculate_newsvendor, NewsvendorInputs
)


# ==============================================================================
# Sample Benchmark Scenarios for Demos & Testing
# ==============================================================================

SAMPLE_QUESTIONS = [
    {
        "id": "eoq",
        "title": "1. Economic Order Quantity (EOQ)",
        "question": (
            "Our annual demand is 12,000 units, ordering cost is $50 per order, "
            "and annual holding cost is $2.00 per unit. We operate 250 working days per year "
            "and lead time is 5 days. What is our optimal order quantity, how often should we order, "
            "and what is our reorder point?"
        ),
        "description": "Determines optimal batch size balancing annual setup and inventory holding costs."
    },
    {
        "id": "discount",
        "title": "2. Quantity Discount Model",
        "question": (
            "We have annual demand of 10,000 units and ordering cost is $100 per order. "
            "Our supplier offers a volume discount schedule with holding cost rate of 20%: "
            "0-999 units at $10.00, 1000-1999 units at $9.80, and 2000+ units at $9.50. "
            "What quantity discount tier should we choose to minimize total annual cost?"
        ),
        "description": "Evaluates all-units price breaks to find optimal order quantity with acquisition costs."
    },
    {
        "id": "safety_stock",
        "title": "3. Safety Stock & Reorder Point",
        "question": (
            "Our daily demand is 100 units with a standard deviation of 20 units. "
            "Supplier lead time is 4 days. We require a 95% cycle service level. "
            "What is our recommended safety stock and reorder point?"
        ),
        "description": "Calculates statistical safety buffer and reorder trigger under continuous review."
    },
    {
        "id": "periodic",
        "title": "4. Fixed-Period Review (Order-Up-To)",
        "question": (
            "We operate a periodic review inventory policy and review inventory every 14 days "
            "with a 7 day supplier lead time. Daily demand is 10 units with a std dev of 3 units. "
            "Service level is 95%. Current inventory on hand is 50 units and on order is 40 units. "
            "What is our order-up-to target level and how much should we order today?"
        ),
        "description": "Determines protection interval buffer and order-up-to replenishment quantity."
    },
    {
        "id": "newsvendor",
        "title": "5. Newsvendor Model (Single-Period)",
        "question": (
            "We sell winter jackets for $100 that cost $40 each. Unsold jackets can be salvaged "
            "for $10 at the end of the season. Seasonal demand is normally distributed with "
            "mean 1000 and standard deviation 200. According to the newsvendor model, "
            "how many jackets should we order and what is our expected profit?"
        ),
        "description": "Solves critical fractile trade-off between underage and overage for seasonal goods."
    }
]


# ==============================================================================
# Terminal CLI Formatting Helpers
# ==============================================================================

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}========================================================================================
     SUPPLY CHAIN ANALYTICS — INVENTORY DECISION AGENT
     Architecture: Question -> Identify -> Extract -> Tool -> Calculate -> Verify -> Explain
========================================================================================{Colors.END}
"""
    print(banner)


def print_step_trace(step_num: int, title: str, summary: str, details: Optional[Dict[str, Any]] = None):
    print(f"{Colors.BLUE}{Colors.BOLD}[Step {step_num}] {title.upper()}{Colors.END}")
    print(f"  {Colors.GREEN}-> {summary}{Colors.END}")
    if details:
        for k, v in details.items():
            if k not in ["outputs_repr", "explanation_length"]:
                print(f"     * {k}: {v}")
    print()


def display_cli_result(result: AgentDecisionResult):
    print_banner()
    print(f"{Colors.YELLOW}{Colors.BOLD}CUSTOMER QUESTION:{Colors.END} \"{result.question}\"\n")
    print(f"{Colors.CYAN}{Colors.BOLD}------------------- 6-STAGE EXECUTION TRACE -------------------{Colors.END}")
    for trace in result.step_traces:
        print_step_trace(trace.step_number, trace.step_name, trace.summary, trace.details)

    print(f"{Colors.CYAN}{Colors.BOLD}------------------- DECISION REPORT & EXPLANATION -------------------{Colors.END}")
    print(result.explanation)
    print(f"\n{Colors.CYAN}{Colors.BOLD}========================================================================================{Colors.END}\n")


# ==============================================================================
# CLI Interactive Loop & Arguments
# ==============================================================================

def run_cli_demo():
    print_banner()
    print(f"{Colors.BOLD}Executing Automated Benchmark Demonstration across all 5 Inventory Models...{Colors.END}\n")
    agent = InventoryDecisionAgent()
    for item in SAMPLE_QUESTIONS:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}>>> Running Scenario: {item['title']}{Colors.END}")
        res = agent.run(item["question"])
        display_cli_result(res)
        print("-" * 88)


def run_cli_interactive():
    agent = InventoryDecisionAgent()
    print_banner()
    print("Welcome to the Supply Chain Analytics Inventory Decision Agent!")
    print("You can type any natural-language inventory question, or choose a benchmark preset.\n")

    while True:
        print(f"{Colors.BOLD}Choose an option:{Colors.END}")
        for i, item in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  [{i}] {item['title']}")
        print("  [6] Enter a custom natural language question")
        print("  [7] Run all 5 model demo scenarios")
        print("  [0] Exit")

        choice = input(f"\n{Colors.GREEN}Enter choice [0-7]: {Colors.END}").strip()

        if choice == "0":
            print("\nExiting Inventory Decision Agent. Goodbye!\n")
            break
        elif choice in ["1", "2", "3", "4", "5"]:
            idx = int(choice) - 1
            sample = SAMPLE_QUESTIONS[idx]
            print(f"\n{Colors.YELLOW}Selected: {sample['title']}{Colors.END}")
            print(f"Question: \"{sample['question']}\"\n")
            res = agent.run(sample["question"])
            display_cli_result(res)
        elif choice == "6":
            custom_q = input(f"\n{Colors.CYAN}Enter your inventory question: {Colors.END}").strip()
            if custom_q:
                res = agent.run(custom_q)
                display_cli_result(res)
        elif choice == "7":
            run_cli_demo()
        else:
            print(f"{Colors.RED}Invalid option. Please choose between 0 and 7.{Colors.END}\n")


# ==============================================================================
# Streamlit Web Application
# ==============================================================================

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown("""
    **Autonomous Decision Intelligence for Supply Chain Operations**  
    *Architecture:* `Customer Question` ➔ `Identify Model` ➔ `Extract Inputs` ➔ `Select Python Tool` ➔ `Calculate` ➔ `Verify` ➔ `Explain`
    """)

    # Sidebar: Model selector & Presets
    st.sidebar.header("🎯 Preset Scenarios")
    preset_choice = st.sidebar.selectbox(
        "Load a Textbook Benchmark Scenario:",
        ["None (Custom Question)"] + [s["title"] for s in SAMPLE_QUESTIONS]
    )

    selected_sample = None
    default_text = ""
    if preset_choice != "None (Custom Question)":
        for s in SAMPLE_QUESTIONS:
            if s["title"] == preset_choice:
                selected_sample = s
                default_text = s["question"]
                break

    # Architecture Overview Expander
    with st.expander("ℹ️ About the 6-Stage Inventory Decision Architecture"):
        st.markdown("""
        1. **Customer Question:** Ingests unconstrained natural-language supply chain queries.
        2. **Identify Model:** Classifies query into EOQ, Quantity Discount, Safety Stock/ROP, Fixed-Period, or Newsvendor.
        3. **Extract Inputs:** High-precision parameter extraction for demand rates, setup costs, lead times, service levels, and price breaks.
        4. **Select Python Tool:** Dynamically binds to pure-Python numerical calculation routines in `inventory_tools.py`.
        5. **Calculate:** Purely deterministic, mathematically rigorous execution.
        6. **Verify:** Mathematical sanity verification (cost parity, convexity, non-negativity, inventory conservation).
        7. **Explain:** Generates executive recommendations, step-by-step math breakdowns, and managerial trade-offs.
        """)

    # Main Input Area
    user_question = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text,
        height=130,
        placeholder="e.g. Our annual demand is 12,000 units, ordering cost is $50, and holding cost is $2/unit/year. What is our optimal order quantity?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        calculate_btn = st.button("🚀 Analyze & Decide", type="primary")

    if calculate_btn and user_question.strip():
        agent = InventoryDecisionAgent()
        with st.spinner("Executing 6-stage analytical pipeline..."):
            result: AgentDecisionResult = agent.run(user_question.strip())

        if not result.success:
            st.error(f"❌ Analysis Incomplete: {result.error_message}")
            if result.step_traces:
                st.subheader("Pipeline Trace")
                for trace in result.step_traces:
                    st.write(f"**Step {trace.step_number}: {trace.step_name}** - {trace.summary}")
            return

        # Top Metric Cards
        st.success(f"✅ Identified Model: **{result.model_type.value}** (Tool: `{result.selected_tool_name}`)")

        kpi_cols = st.columns(4)
        if result.model_type == InventoryModelType.EOQ:
            kpi_cols[0].metric("Optimal Order (Q*)", f"{result.raw_outputs.optimal_order_quantity:,.1f} units")
            kpi_cols[1].metric("Annual Orders", f"{result.raw_outputs.annual_orders_count:.1f} / year")
            kpi_cols[2].metric("Total Relevant Cost", f"${result.raw_outputs.total_relevant_cost:,.2f}")
            kpi_cols[3].metric("Order Cycle", f"{result.raw_outputs.order_cycle_days:.1f} days")
        elif result.model_type == InventoryModelType.QUANTITY_DISCOUNT:
            kpi_cols[0].metric("Recommended Tier", result.raw_outputs.optimal_tier_label)
            kpi_cols[1].metric("Optimal Batch", f"{result.raw_outputs.optimal_order_quantity:,.0f} units")
            kpi_cols[2].metric("Unit Price", f"${result.raw_outputs.optimal_unit_price:,.2f}")
            kpi_cols[3].metric("Annual Savings", f"${result.raw_outputs.annual_cost_savings:,.2f}", f"{result.raw_outputs.percentage_savings:.1f}%")
        elif result.model_type == InventoryModelType.SAFETY_STOCK_ROP:
            kpi_cols[0].metric("Safety Stock (SS)", f"{result.raw_outputs.safety_stock:,.1f} units")
            kpi_cols[1].metric("Reorder Point (ROP)", f"{result.raw_outputs.reorder_point:,.1f} units")
            kpi_cols[2].metric("Service Level", f"{result.raw_outputs.service_level*100:.1f}%")
            kpi_cols[3].metric("Z-Score", f"{result.raw_outputs.z_score:.2f}")
        elif result.model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            kpi_cols[0].metric("Order-Up-To Level (M)", f"{result.raw_outputs.target_inventory_level:,.1f} units")
            kpi_cols[1].metric("Safety Stock", f"{result.raw_outputs.safety_stock:,.1f} units")
            kpi_cols[2].metric("Protection Interval", f"{result.raw_outputs.protection_interval:.0f} periods")
            if result.raw_outputs.recommended_order_quantity is not None:
                kpi_cols[3].metric("Order Today", f"{result.raw_outputs.recommended_order_quantity:,.1f} units")
        elif result.model_type == InventoryModelType.NEWSVENDOR:
            kpi_cols[0].metric("Optimal Order (Q*)", f"{result.raw_outputs.optimal_order_quantity:,.0f} units")
            kpi_cols[1].metric("Critical Ratio", f"{result.raw_outputs.critical_ratio*100:.2f}%")
            kpi_cols[2].metric("Expected Profit", f"${result.raw_outputs.expected_profit:,.2f}")
            kpi_cols[3].metric("EVPI", f"${result.raw_outputs.evpi:,.2f}")

        # Tabs for Decision Report & Technical Pipeline Trace
        tab_report, tab_trace, tab_verify = st.tabs(["📊 Decision Report & Explanation", "🔍 6-Stage Pipeline Trace", "✅ Verification Report"])

        with tab_report:
            st.markdown(result.explanation)

        with tab_trace:
            st.subheader("Step-by-Step Architecture Trace")
            for t in result.step_traces:
                with st.expander(f"Step {t.step_number}: {t.step_name} — {t.summary}", expanded=(t.step_number <= 3)):
                    st.json(t.details)

        with tab_verify:
            st.subheader("Mathematical Constraints & Quality Verification")
            for c in result.verification_report.get("checks", []):
                if c["passed"]:
                    st.success(f"**PASS:** {c['check']} — *{c['detail']}*")
                else:
                    st.error(f"**FAIL:** {c['check']} — *{c['detail']}*")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    # Detect if invoked under 'streamlit run'
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            run_streamlit_app()
            return
    except ImportError:
        pass
    except Exception:
        pass

    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("--question", "-q", type=str, help="Natural language inventory question to process")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demo across all 5 inventory models")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    if args.streamlit:
        import subprocess
        subprocess.run(["streamlit", "run", __file__])
    elif args.demo:
        run_cli_demo()
    elif args.question:
        agent = InventoryDecisionAgent()
        res = agent.run(args.question)
        display_cli_result(res)
    else:
        run_cli_interactive()


if __name__ == "__main__":
    main()
