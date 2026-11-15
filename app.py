"""Supply Chain Analytics Inventory Decision Agent Application

Dual-mode interface:
1. Streamlit Interactive Web Application (run with `streamlit run app.py`)
2. Command-Line Interface (run with `python app.py`)

Follows the 6-stage decision architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from inventory_agent import InventoryDecisionAgent, ModelType


PRESET_SCENARIOS = {
    "1. Economic Order Quantity (EOQ)": (
        "We are managing an inventory of industrial ball bearings. "
        "Our annual demand is 12,000 units per year. "
        "It costs $60 to place each order and our holding cost is $2.50 per unit per year. "
        "What is our optimal Economic Order Quantity and how often should we order?"
    ),
    "2. Quantity Discount": (
        "Our retail distribution warehouse has an annual demand of 5,000 cases. "
        "Each order incurs an administrative cost of $49, and our annual holding cost rate is 20% of unit price. "
        "Our supplier offers the following quantity discounts: "
        "1 to 999 units at $5.00, 1000 or more units at $4.80. "
        "Which price tier should we select, and how much should we order to meet our demand?"
    ),
    "3. Safety Stock and Reorder Point (Continuous Review)": (
        "We need to configure our replenishment alert policy for medical surgical gloves. "
        "Supplier lead time is 5 days. Average daily demand is 80 boxes with a standard deviation of 12 boxes per day. "
        "Our target cycle service level is 98%. "
        "What is our required safety stock and reorder point?"
    ),
    "4. Fixed-Period Review (Order-Up-To Policy)": (
        "We manage fresh organic produce on a periodic review schedule. "
        "We review inventory every 7 days (review period = 7 days) and supplier lead time is 3 days. "
        "Average demand is 150 units per day with a standard deviation of 25 units per day. "
        "Target service level is 95%. "
        "Currently we have 400 units on hand, 250 units in transit (on order), and zero backorders. "
        "What is our order-up-to target level and how many units should we order today?"
    ),
    "5. Newsvendor (Single-Period Probabilistic)": (
        "We are ordering commemorative athletic jerseys for a one-weekend championship event under the newsvendor model. "
        "The selling price is $120 per jersey, procurement cost is $45, and any leftover jerseys after the event can be salvaged for $15. "
        "Demand is normally distributed with mean 2,500 units and standard deviation 400 units. "
        "What is the optimal quantity to stock and what is our expected profit?"
    ),
}


# ==============================================================================
# Streamlit Web App Mode
# ==============================================================================

def render_dropdown_decision_report(response, default_expanded: bool = False):
    import re
    import streamlit as st

    # 1. Core Decision Callout Banner at the top
    recommendation = ""
    if response.raw_tool_output and "recommendation" in response.raw_tool_output:
        recommendation = response.raw_tool_output["recommendation"]
    elif "> **Core Decision**:" in response.explanation:
        m = re.search(r">\s*\*\*Core Decision\*\*:\s*([^\n]+)", response.explanation)
        if m:
            recommendation = m.group(1).strip()

    st.subheader(f"📋 {response.model_name}")
    if recommendation:
        st.success(f"🎯 **Core Decision**: {recommendation}")

    # Section mapping with icons and titles
    step_icons = {
        "1": "📥",
        "2": "📖",
        "3": "📐",
        "4": "🔢",
        "5": "🧮",
        "6": "🏁",
        "7": "🛡️",
        "8": "📝",
        "9": "💡",
    }

    # Split markdown by '### '
    sections = re.split(r"\n+###\s+", response.explanation)

    for s in sections[1:]:
        lines = s.split("\n", 1)
        raw_title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        body = re.sub(r"\n*---+\s*$", "", body).strip()

        # Extract step number if present, e.g. "1. Inputs Extracted" -> "1"
        num_m = re.match(r"^(\d+)\.\s*(.*)", raw_title)
        if num_m:
            step_num = num_m.group(1)
            step_name = num_m.group(2)
            icon = step_icons.get(step_num, "🔹")
            expander_title = f"{icon} Step {step_num}: {step_name}"
        else:
            expander_title = f"🔹 {raw_title}"

        with st.expander(expander_title, expanded=default_expanded):
            st.markdown(body)


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Analytics: Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.caption(
        "Autonomous Decision System implementing the 6-stage pipeline: "
        "**Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain**"
    )

    # Sidebar: Model Directory & Presets
    with st.sidebar:
        st.header("⚙️ Configuration & Scenarios")
        selected_scenario_label = st.selectbox(
            "Select a Preset Inventory Scenario:",
            ["Custom Question"] + list(PRESET_SCENARIOS.keys()),
        )

        st.markdown("---")
        st.subheader("📚 Supported Inventory Models")
        st.markdown(
            """
            1. **Economic Order Quantity (EOQ)**
               - Wilson formulation ($Q^* = \\sqrt{2DS/H}$)
            2. **Quantity Discount**
               - Multi-tier price-break optimization
            3. **Safety Stock & Reorder Point**
               - Continuous review $(s, Q)$ with stochastic demand/lead time
            4. **Fixed-Period Review**
               - Periodic review order-up-to policy $(R, S)$
            5. **Newsvendor Model**
               - Single-period critical fractile ($C_u / (C_u + C_o)$)
            """
        )

    # Input question container
    if selected_scenario_label != "Custom Question":
        default_question = PRESET_SCENARIOS[selected_scenario_label]
    else:
        default_question = (
            "Annual demand is 10000 units, order setup cost is $50, and holding cost is $2 per unit per year. "
            "What is our EOQ?"
        )

    st.subheader("💬 Ask Your Inventory Question")
    question_input = st.text_area(
        "Enter your question in natural language (mention demand, costs, lead time, review periods, etc.):",
        value=default_question,
        height=110,
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_button = st.button("🚀 Analyze & Decide", type="primary", use_container_width=True)

    if run_button or "agent_response" not in st.session_state:
        agent = InventoryDecisionAgent()
        with st.spinner("Analyzing inventory question through the 6-stage decision pipeline..."):
            response = agent.run(question_input)
            st.session_state["agent_response"] = response

    response = st.session_state.get("agent_response")

    if response:
        st.markdown("---")
        st.header("🔍 Pipeline Execution Traces")

        # Display Pipeline Stage Pills / Cards
        stage_cols = st.columns(6)
        stages_info = [
            ("1. Identify Model", "🎯", response.model_type.value if response.model_type else "Pending"),
            ("2. Extract Inputs", "📋", f"{len(response.extracted_inputs)} Extracted"),
            ("3. Select Tool", "🛠️", response.selected_tool_name.split(".")[-1] if response.selected_tool_name else "N/A"),
            ("4. Calculate", "🧮", "Executed" if response.calculation_results else "Error"),
            ("5. Verify", "🛡️", "Passed" if (response.verification and response.verification.get("is_valid")) else "Check"),
            ("6. Explain", "💡", "Ready"),
        ]

        for col, (name, icon, status) in zip(stage_cols, stages_info):
            with col:
                st.metric(label=f"{icon} {name}", value=status)

        col_heading, col_toggle = st.columns([4, 1])
        with col_toggle:
            expand_all = st.checkbox("📂 Expand all steps", value=False, help="Click to expand or collapse all step dropdown menus")

        # Tabs for deep dive
        tab_report, tab_stages, tab_results = st.tabs(["📄 Decision Report", "🔄 Architecture Traces", "📊 Raw Results & Checks"])

        with tab_report:
            if response.success:
                render_dropdown_decision_report(response, default_expanded=expand_all)
            else:
                st.error(f"**Pipeline Stopped**: {response.error_message}")
                with st.expander("📋 Step 1: Extracted Parameters", expanded=True):
                    if response.extracted_inputs:
                        for k, v in response.extracted_inputs.items():
                            st.write(f"- **{k}**: `{v}`")
                    else:
                        st.write("*(No parameters could be extracted)*")
                with st.expander("⚠️ Step 2: Missing Required Parameters", expanded=True):
                    for m in response.missing_inputs:
                        st.markdown(f"- ⚠️ **{m}**")
                    st.info("Please provide the missing parameter values in your question to calculate the inventory decision.")

        with tab_stages:
            st.caption("Click on each pipeline step below to expand its execution details:")
            for trace in response.stage_traces:
                with st.expander(f"📍 **{trace.stage_name}**: {trace.summary}", expanded=expand_all):
                    st.json(trace.details)

        with tab_results:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("Extracted Inputs")
                st.json(response.extracted_inputs)
                if response.missing_inputs:
                    st.warning(f"Missing Inputs: {response.missing_inputs}")
            with col_res2:
                st.subheader("Calculation Metrics")
                if response.calculation_results:
                    st.json(response.calculation_results)
                else:
                    st.write("No metrics computed.")


# ==============================================================================
# CLI Mode
# ==============================================================================

def run_cli_interactive():
    agent = InventoryDecisionAgent()
    print("=" * 78)
    print("  SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT (CLI Mode)")
    print("=" * 78)
    print("Supports: 1) EOQ  2) Quantity Discount  3) Safety Stock & ROP")
    print("          4) Fixed-Period Review  5) Newsvendor")
    print("Commands: Enter a number 1-5 to run a preset scenario, or type your question.")
    print("          Type 'exit' or 'quit' to end.\n")

    scenario_keys = list(PRESET_SCENARIOS.keys())

    while True:
        try:
            user_input = input("\n[Customer Question] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting Supply Chain Decision Agent. Goodbye!")
            break

        # Check for numeric shortcuts
        if user_input in ("1", "2", "3", "4", "5"):
            chosen_key = scenario_keys[int(user_input) - 1]
            question = PRESET_SCENARIOS[chosen_key]
            print(f"\nRunning Scenario: {chosen_key}")
            print(f"Question: {question}\n")
        else:
            question = user_input

        print("\n" + "-" * 78)
        print("  Executing 6-Stage Inventory Decision Pipeline...")
        print("-" * 78)

        response = agent.run(question)

        # Print stage traces
        for trace in response.stage_traces:
            print(f"  • {trace.stage_name:24s} -> {trace.summary}")

        print("-" * 78)

        if response.success:
            print(f"\n{response.explanation}\n")
        else:
            print(f"\n[ALERT] Pipeline stopped: {response.error_message}")
            print(f"{response.explanation}\n")


def main():
    # If streamlit is invoked (e.g. `streamlit run app.py`), sys.argv will typically contain streamlit
    # or streamlit runtime is loaded
    is_streamlit_running = False
    try:
        import streamlit as st
        # In streamlit runtime, st.runtime.exists() evaluates to True
        if hasattr(st, "runtime") and st.runtime.exists():
            is_streamlit_running = True
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("--cli", action="store_true", help="Force interactive CLI mode")
    parser.add_argument("--question", type=str, help="Single question to evaluate directly in terminal")
    args, _ = parser.parse_known_args()

    if args.question:
        agent = InventoryDecisionAgent()
        res = agent.run(args.question)
        print(res.explanation)
        return

    if is_streamlit_running and not args.cli:
        run_streamlit_app()
    else:
        run_cli_interactive()


if __name__ == "__main__":
    main()