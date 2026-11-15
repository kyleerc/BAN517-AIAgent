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


def render_csv_data_analysis(uploaded_file):
    import pandas as pd
    import streamlit as st
    from inventory_tools import classify_inventory_multicriteria

    st.markdown("---")

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return

    # Schema detection
    is_classification_schema = any(
        c in df.columns for c in ["Sales ($)", "Sales", "sales", "SALES", "Sales_Dollar", "Revenue"]
    ) and any(c in df.columns for c in ["POS Hits", "pos hits", "POS_Hits", "Hits", "hits", "HITS", "Hit", "Picks"])

    is_basic_inventory_schema = all(
        c in df.columns for c in ["Product", "Category", "Units", "Unit Cost"]
    )

    if not is_classification_schema and not is_basic_inventory_schema:
        st.error("⚠️ Uploaded CSV does not match either supported inventory schema:")
        st.markdown(
            """
            1. **Multi-Criteria Inventory Classification**: Requires columns `Item Number`, `Item Name`, `UoM`, `Sales ($)` (or `Sales`), and `POS Hits` (or `Hits`).
            2. **Inventory Value Analysis**: Requires columns `Product`, `Category`, `Units`, and `Unit Cost`.
            """
        )
        st.caption("Uploaded Columns: " + ", ".join(list(df.columns)))
        st.dataframe(df, use_container_width=True)
        return

    # ==========================================================================
    # Capability 1: Multi-Criteria Inventory Classification (Sales & POS Hits)
    # ==========================================================================
    if is_classification_schema:
        st.header("🏷️ Multi-Criteria Inventory Classification (A, B, C, D)")
        st.caption(
            "Classification pipeline: "
            "**Step 1:** Pareto Ranking (Sales 0-60% A, 61-80% B, 81-90% C, 91-100% D; POS Hits 0-60% A, 61-80% B, 81-90% C, 91-100% D) → "
            "Scores (A=4, B=3, C=2, D=1) | "
            "**Step 2:** Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score) | "
            "**Step 3:** Final Class (A: 4.00-3.25, B: 3.24-2.50, C: 2.49-1.75, D: < 1.75)"
        )

        try:
            res = classify_inventory_multicriteria(df)
        except Exception as e:
            st.error(f"Classification failed: {e}")
            return

        classified_df = res["classified_df"]
        summary_df = res["class_summary"]
        crosstab_df = res["crosstab"]
        total_skus = res["total_skus"]
        total_sales = res["total_sales"]
        total_hits = res["total_hits"]
        sales_col = res["sales_col"]
        hits_col = res["hits_col"]

        # Dataset Overview
        num_rows = len(df)
        num_cols = len(classified_df.columns)

        meta1, meta2, meta3 = st.columns(3)
        with meta1:
            st.metric("Total Items (SKUs)", f"{num_rows:,}")
        with meta2:
            st.metric("Total Columns", f"{num_cols}")
        with meta3:
            st.markdown(f"**Calculated Fields:**\n`Sales Score`, `Hits Score`, `Combined Weighted Score`, `Class`")

        # Top Executive KPIs
        st.subheader("📈 Classification Summary Metrics")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.metric("Total Sales ($)", f"${total_sales:,.2f}")
        with kpi2:
            st.metric("Total POS Hits", f"{total_hits:,.0f}")
        with kpi3:
            cnt_a = len(classified_df[classified_df["Class"] == "A"])
            st.metric("Category A Items", f"{cnt_a:,}", delta=f"{cnt_a/total_skus*100:.1f}%")
        with kpi4:
            cnt_b = len(classified_df[classified_df["Class"] == "B"])
            st.metric("Category B Items", f"{cnt_b:,}", delta=f"{cnt_b/total_skus*100:.1f}%")
        with kpi5:
            cnt_cd = len(classified_df[classified_df["Class"].isin(["C", "D"])])
            st.metric("Category C & D Items", f"{cnt_cd:,}", delta=f"{cnt_cd/total_skus*100:.1f}%")

        # Step 4: Required Classification Summary
        st.subheader("📋 Step 4: Required Classification Summary")
        st.markdown(
            "Summary reporting the **number of items in each final category (A, B, C, and D)** "
            "and the **total Sales** for all items in each category:"
        )

        tab_dist, tab_matrix = st.tabs(["📊 Category Summary Table & Visuals", "🔲 Bi-Criteria Matrix Grid"])

        with tab_dist:
            col_t1, col_t2 = st.columns([3, 2])
            with col_t1:
                st.dataframe(
                    summary_df,
                    column_config={
                        "Final Category": st.column_config.TextColumn("Final Category"),
                        "Item Count": st.column_config.NumberColumn("Item Count", format="%d"),
                        "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
                        "% of Sales": st.column_config.NumberColumn("% of Sales", format="%.1f%%"),
                        "Total POS Hits": st.column_config.NumberColumn("Total POS Hits", format="%d"),
                        "% of POS Hits": st.column_config.NumberColumn("% of POS Hits", format="%.1f%%"),
                        "% of Items": st.column_config.NumberColumn("% of Items", format="%.1f%%"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
            with col_t2:
                st.bar_chart(summary_df.set_index("Final Category")[["Total Sales ($)"]])

        with tab_matrix:
            st.markdown("**Bi-Criteria Cross-Tabulation Grid (Sales Class vs. POS Hits Class)**:")
            st.dataframe(crosstab_df, use_container_width=True)

        # Classified Data Table with interactive filtering
        st.subheader("📋 Classified Inventory Data Table")
        filter_col, dl_col = st.columns([2, 1])
        with filter_col:
            selected_class = st.selectbox(
                "Filter table by Final Inventory Class:",
                ["All Categories (A, B, C, D)", "Category A", "Category B", "Category C", "Category D"],
            )
        with dl_col:
            st.write("")
            st.download_button(
                "📥 Download Classified CSV",
                data=classified_df.to_csv(index=False),
                file_name="classified_inventory.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if selected_class.startswith("Category "):
            filtered_df = classified_df[classified_df["Class"] == selected_class.split()[-1]]
        else:
            filtered_df = classified_df

        st.dataframe(
            filtered_df,
            column_config={
                sales_col: st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                "Sales Cumulative %": st.column_config.NumberColumn("Sales Cum %", format="%.2f%%"),
                hits_col: st.column_config.NumberColumn("POS Hits", format="%d"),
                "Hits Cumulative %": st.column_config.NumberColumn("Hits Cum %", format="%.2f%%"),
                "Combined Weighted Score": st.column_config.NumberColumn("Combined Weighted Score", format="%.2f"),
            },
            use_container_width=True,
        )

    # ==========================================================================
    # Capability 2: Existing Basic Inventory Value Analysis
    # ==========================================================================
    if is_basic_inventory_schema:
        st.header("📊 Inventory Data Analysis")

        # Clean numeric columns if strings contain currency symbols or commas
        if df["Units"].dtype == object:
            df["Units"] = pd.to_numeric(df["Units"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        else:
            df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

        if df["Unit Cost"].dtype == object:
            df["Unit Cost"] = pd.to_numeric(
                df["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            )
        else:
            df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce")

        num_rows = len(df)
        num_cols = len(df.columns)
        col_names_str = ", ".join(df.columns.tolist())

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("Number of Rows", f"{num_rows:,}")
        with col_meta2:
            st.metric("Number of Columns", f"{num_cols}")
        with col_meta3:
            st.markdown(f"**Column Names:**\n`{col_names_str}`")

        # 1. Total Units
        total_units = df["Units"].sum()

        # 2. Average Unit Cost
        avg_unit_cost = df["Unit Cost"].mean()

        # 3. Product with highest Units
        if not df.empty and df["Units"].notna().any():
            highest_units_idx = df["Units"].idxmax()
            highest_product = df.loc[highest_units_idx, "Product"]
            highest_units_val = df.loc[highest_units_idx, "Units"]
        else:
            highest_product = "N/A"
            highest_units_val = 0

        # 4. Group data by Category and calculate total Units for each Category
        category_units = df.groupby("Category")["Units"].sum().reset_index()

        # 5. Create new calculated column: Inventory Value = Units × Unit Cost
        df["Inventory Value"] = df["Units"] * df["Unit Cost"]

        # 6. Calculate total Inventory Value across all products
        total_inventory_value = df["Inventory Value"].sum()

        # RESULTS display
        st.subheader("📈 Summary Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Units", f"{total_units:,.0f}")
        with m2:
            st.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
        with m3:
            st.metric(
                "Product with Highest Units",
                f"{highest_product}",
                delta=f"{highest_units_val:,.0f} units" if highest_product != "N/A" else None,
            )
        with m4:
            st.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

        # Total Units by Category
        st.subheader("📂 Total Units by Category")
        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.dataframe(
                category_units,
                column_config={
                    "Units": st.column_config.NumberColumn("Total Units", format="%d")
                },
                use_container_width=True,
                hide_index=True,
            )
        with c_col2:
            st.bar_chart(category_units.set_index("Category")["Units"])

        # Display the dataframe including the new Inventory Value column
        st.subheader("📋 Inventory Data Table (with Inventory Value)")
        st.dataframe(
            df,
            column_config={
                "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                "Units": st.column_config.NumberColumn("Units", format="%d"),
            },
            use_container_width=True,
        )


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

        st.markdown("---")
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload an inventory CSV file with Product, Category, Units, and Unit Cost columns.",
        )

    # Optional CSV Data Analysis View
    if uploaded_file is not None:
        render_csv_data_analysis(uploaded_file)

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