"""Deterministic Supply Chain Analytics Inventory Calculation Tools.

Implements pedagogical, verify-ready calculation engines for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (Safety Stock and Reorder Point)
4. calculate_periodic_review() (Fixed-Period Review)
5. calculate_newsvendor()

Every tool strictly returns:
1. Inputs Extracted (inputs)
2. Variable Definitions (variable_definitions)
3. Formula (formula)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

Constraints:
- Strict validation of all inputs (types, bounds, domain rules).
- Never invent missing numerical inputs (raises InventoryCalculationError).
- Pure deterministic Python calculations without external approximations.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union


class InventoryCalculationError(ValueError):
    """Raised when numerical calculation parameters violate domain boundaries or are missing."""
    pass


# ==============================================================================
# 1. Economic Order Quantity (EOQ)
# ==============================================================================

def calculate_eoq(
    demand: float,
    order_cost: float,
    holding_cost: float,
    unit_cost: Optional[float] = None,
    workdays_per_year: int = 250,
) -> Dict[str, Any]:
    """Calculate the Economic Order Quantity (Wilson EOQ model).

    Formula:
        Q* = sqrt((2 * D * S) / H)

    Returns all 8 verification sections for student learning.
    """
    # 1. Input Validation (never invent missing numerical inputs)
    if demand is None:
        raise InventoryCalculationError("Annual demand 'demand' is required and cannot be None.")
    if order_cost is None:
        raise InventoryCalculationError("Order cost 'order_cost' is required and cannot be None.")
    if holding_cost is None:
        raise InventoryCalculationError("Holding cost 'holding_cost' is required and cannot be None.")

    try:
        d = float(demand)
        s = float(order_cost)
        h = float(holding_cost)
        c = float(unit_cost) if unit_cost is not None else None
        workdays = int(workdays_per_year)
    except (ValueError, TypeError) as err:
        raise InventoryCalculationError(f"Numerical inputs must be convertible to float. Error: {err}")

    if d <= 0:
        raise InventoryCalculationError(f"Annual demand must be > 0. Received: {d}")
    if s <= 0:
        raise InventoryCalculationError(f"Order cost must be > 0. Received: {s}")
    if h <= 0:
        raise InventoryCalculationError(f"Holding cost must be > 0. Received: {h}")
    if c is not None and c <= 0:
        raise InventoryCalculationError(f"Unit cost must be > 0 when specified. Received: {c}")
    if workdays <= 0:
        raise InventoryCalculationError(f"Workdays per year must be > 0. Received: {workdays}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand (D)": d,
        "order_cost (S)": s,
        "holding_cost (H)": h,
        "unit_cost (C)": c,
        "workdays_per_year": workdays,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Fixed setup or ordering cost incurred per replenishment order ($/order)",
        "H": "Annual inventory holding/carrying cost per unit per year ($/unit/year)",
        "C": "Unit purchase acquisition cost ($/unit)",
        "workdays_per_year": "Operating workdays in one business year (days/year)",
        "Q*": "Economic Order Quantity minimizing total relevant inventory costs (units/order)",
        "N": "Order frequency / replenishment orders placed per year (orders/year)",
        "T": "Order cycle length / interval between consecutive orders (workdays)",
        "AOC": "Annual Ordering Cost = N * S ($/year)",
        "AHC": "Annual Holding Cost = (Q* / 2) * H ($/year)",
        "TRC": "Total Relevant Inventory Cost = AOC + AHC ($/year)",
        "TAC": "Total Annual Cost including unit purchase cost = D * C + TRC ($/year)",
    }

    # Section 3: Formula (Symbolic)
    formula = {
        "EOQ": "Q* = sqrt((2 * D * S) / H)",
        "Orders per Year (N)": "N = D / Q*",
        "Order Cycle Interval (T)": "T = (Q* / D) * workdays_per_year",
        "Annual Ordering Cost (AOC)": "AOC = (D / Q*) * S",
        "Annual Holding Cost (AHC)": "AHC = (Q* / 2) * H",
        "Total Relevant Cost (TRC)": "TRC = AOC + AHC = (D / Q*) * S + (Q* / 2) * H",
        "Total Annual Cost (TAC)": "TAC = D * C + TRC",
    }

    # Deterministic Intermediate Calculations
    numerator = 2.0 * d * s
    ratio = numerator / h
    eoq = math.sqrt(ratio)
    orders_per_year = d / eoq
    cycle_time_years = eoq / d
    cycle_time_days = cycle_time_years * workdays

    annual_ordering_cost = orders_per_year * s
    annual_holding_cost = (eoq / 2.0) * h
    total_relevant_cost = annual_ordering_cost + annual_holding_cost
    annual_purchase_cost = (d * c) if c is not None else None
    total_annual_cost = (total_relevant_cost + annual_purchase_cost) if annual_purchase_cost is not None else total_relevant_cost

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = {
        "EOQ Substitution": (
            f"Q* = sqrt((2 * {d:,.2f} * {s:,.2f}) / {h:,.2f}) = "
            f"sqrt({numerator:,.2f} / {h:,.2f}) = sqrt({ratio:,.2f}) = {eoq:,.2f} units"
        ),
        "Orders per Year Substitution": f"N = {d:,.2f} / {eoq:,.2f} = {orders_per_year:,.2f} orders/year",
        "Cycle Interval Substitution": f"T = ({eoq:,.2f} / {d:,.2f}) * {workdays} days = {cycle_time_days:,.1f} workdays",
        "Annual Ordering Cost Substitution": f"AOC = {orders_per_year:,.2f} * ${s:,.2f} = ${annual_ordering_cost:,.2f}/year",
        "Annual Holding Cost Substitution": f"AHC = ({eoq:,.2f} / 2) * ${h:,.2f} = ${annual_holding_cost:,.2f}/year",
        "Total Relevant Cost Substitution": f"TRC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_relevant_cost:,.2f}/year",
    }
    if annual_purchase_cost is not None:
        formula_with_substituted_values["Total Annual Cost Substitution"] = (
            f"TAC = ({d:,.2f} * ${c:,.2f}) + ${total_relevant_cost:,.2f} = "
            f"${annual_purchase_cost:,.2f} + ${total_relevant_cost:,.2f} = ${total_annual_cost:,.2f}/year"
        )

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "numerator_2DS": numerator,
        "ratio_2DS_over_H": ratio,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "ordering_holding_cost_difference": abs(annual_ordering_cost - annual_holding_cost),
        "total_relevant_cost": total_relevant_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
    }

    # Section 6: Final Answer
    final_answer = {
        "optimal_order_quantity": eoq,
        "recommended_order_quantity_int": int(round(eoq)),
        "orders_per_year": orders_per_year,
        "order_interval_workdays": cycle_time_days,
        "total_relevant_cost": total_relevant_cost,
        "total_annual_cost": total_annual_cost,
    }

    # Section 7: Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    is_cost_balanced = cost_diff < 1e-4 or (cost_diff / total_relevant_cost) < 1e-4
    second_derivative = (2.0 * d * s) / (eoq ** 3)
    verification_checks = [
        {
            "check": "Positivity Constraint",
            "passed": eoq > 0 and d > 0,
            "details": f"EOQ={eoq:.2f} units (> 0)",
        },
        {
            "check": "First-Order Optimality (AOC ≈ AHC at minimum)",
            "passed": is_cost_balanced,
            "details": f"AOC (${annual_ordering_cost:.2f}) vs AHC (${annual_holding_cost:.2f}), Diff=${cost_diff:.4f}",
        },
        {
            "check": "Second-Order Convexity / Global Minimality (d2TC/dQ2 > 0)",
            "passed": second_derivative > 0,
            "details": f"Second derivative = (2*D*S)/(Q^3) = {second_derivative:.6f} > 0 (Strict Global Minimum)",
        },
    ]

    # Section 8: Assumptions
    assumptions = [
        "Demand rate is constant, continuous, and completely known throughout the entire planning year.",
        "Order setup cost is fixed and incurred in full whenever an order batch is placed, regardless of batch size.",
        "Holding cost per unit per year is linear, constant, and independent of order quantity.",
        "Replenishment is instantaneous (delivery lead time is zero or constant and fully predictable).",
        "The entire replenishment order lot arrives in a single batch delivery (no partial shipments).",
        "No shortages, stockouts, or backorders are permitted.",
        "Unit purchase price is constant and independent of order lot size (no quantity price breaks).",
    ]

    steps = [
        f"1. Annual Demand (D) = {d:,.2f} units/year",
        f"2. Order Setup Cost (S) = ${s:,.2f} per order",
        f"3. Holding Cost (H) = ${h:,.2f} per unit/year",
        f"4. EOQ Formula: Q* = sqrt((2 * D * S) / H)",
        f"   Numerator = 2 * {d:,.2f} * {s:,.2f} = {numerator:,.2f}",
        f"   Ratio = {numerator:,.2f} / {h:,.2f} = {ratio:,.2f}",
        f"   Q* = sqrt({ratio:,.2f}) = {eoq:,.2f} units",
        f"5. Orders per Year: N = D / Q* = {d:,.2f} / {eoq:,.2f} = {orders_per_year:,.2f} orders",
        f"6. Order Interval: T = (Q* / D) * {workdays} days = {cycle_time_days:,.1f} workdays",
        f"7. Annual Ordering Cost = N * S = {orders_per_year:,.2f} * ${s:,.2f} = ${annual_ordering_cost:,.2f}",
        f"8. Annual Holding Cost = (Q* / 2) * H = ({eoq:,.2f} / 2) * ${h:,.2f} = ${annual_holding_cost:,.2f}",
        f"9. Total Relevant Inventory Cost = ${total_relevant_cost:,.2f}",
    ]
    if annual_purchase_cost is not None:
        steps.append(f"10. Total Annual Cost (with ${c:,.2f}/unit purchase) = ${total_annual_cost:,.2f}")

    recommendation = (
        f"Order exactly {round(eoq):,} units (or {eoq:.1f} units) per batch. "
        f"This yields {orders_per_year:.1f} orders per year (every {cycle_time_days:.0f} workdays) "
        f"and minimizes total annual holding and ordering costs at ${total_relevant_cost:,.2f}."
    )

    return {
        "tool_name": "calculate_eoq",
        "model_name": "Economic Order Quantity (EOQ)",
        # The 8 Student Verification Sections:
        "inputs": inputs_extracted,
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Standard auxiliary metadata
        "step_by_step": steps,
        "results": {
            "optimal_order_quantity": eoq,
            "recommended_order_quantity_int": int(round(eoq)),
            **intermediate_calculations,
        },
        "verification": {
            "is_valid": all(c["passed"] for c in verification_checks),
            "checks": verification_checks,
        },
        "recommendation": recommendation,
    }


# ==============================================================================
# 2. Quantity Discount Model
# ==============================================================================

def calculate_quantity_discount(
    demand: float,
    order_cost: float,
    tiers: List[Dict[str, Any]],
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate the optimal order quantity under all-units quantity discounts.

    Evaluates feasible order quantities across all discount brackets and selects
    the tier minimizing Total Annual Cost (TAC).

    Returns all 8 verification sections for student learning.
    """
    # 1. Input Validation (never invent missing numerical inputs)
    if demand is None:
        raise InventoryCalculationError("Annual demand 'demand' is required and cannot be None.")
    if order_cost is None:
        raise InventoryCalculationError("Order cost 'order_cost' is required and cannot be None.")
    if tiers is None or not isinstance(tiers, list) or len(tiers) == 0:
        raise InventoryCalculationError("Tiers 'tiers' must be a non-empty list of discount brackets.")
    if holding_cost_rate is None and fixed_holding_cost is None:
        raise InventoryCalculationError(
            "Must provide either 'holding_cost_rate' or 'fixed_holding_cost'; missing holding cost cannot be assumed."
        )

    try:
        d = float(demand)
        s = float(order_cost)
        h_rate = float(holding_cost_rate) if holding_cost_rate is not None else None
        fixed_h = float(fixed_holding_cost) if fixed_holding_cost is not None else None
    except (ValueError, TypeError) as err:
        raise InventoryCalculationError(f"Numerical inputs must be convertible to float. Error: {err}")

    if d <= 0:
        raise InventoryCalculationError(f"Annual demand must be > 0. Received: {d}")
    if s <= 0:
        raise InventoryCalculationError(f"Order cost must be > 0. Received: {s}")
    if h_rate is not None and h_rate <= 0:
        raise InventoryCalculationError(f"Holding cost rate must be > 0. Received: {h_rate}")
    if fixed_h is not None and fixed_h <= 0:
        raise InventoryCalculationError(f"Fixed holding cost must be > 0. Received: {fixed_h}")

    validated_tiers = []
    for i, t in enumerate(tiers, start=1):
        if not isinstance(t, dict):
            raise InventoryCalculationError(f"Tier {i} must be a dictionary. Received: {type(t)}")
        if "unit_price" not in t:
            raise InventoryCalculationError(f"Tier {i} missing required key 'unit_price'.")
        p = float(t["unit_price"])
        if p <= 0:
            raise InventoryCalculationError(f"Unit price for tier {i} must be > 0. Received: {p}")
        min_q = float(t.get("min_qty", 1.0))
        max_q = float(t["max_qty"]) if t.get("max_qty") is not None else None
        if min_q < 0:
            raise InventoryCalculationError(f"Tier {i} min_qty cannot be negative. Received: {min_q}")
        if max_q is not None and max_q < min_q:
            raise InventoryCalculationError(f"Tier {i} max_qty ({max_q}) cannot be less than min_qty ({min_q}).")
        validated_tiers.append({"min_qty": min_q, "max_qty": max_q, "unit_price": p})

    sorted_tiers = sorted(validated_tiers, key=lambda x: x["min_qty"])

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand (D)": d,
        "order_cost (S)": s,
        "holding_cost_rate": h_rate,
        "fixed_holding_cost": fixed_h,
        "tiers": sorted_tiers,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual customer demand (units/year)",
        "S": "Fixed order setup cost ($/order)",
        "j": "Discount tier index (1, 2, ...)",
        "min_qty_j": "Minimum lot size required to qualify for discount tier j",
        "max_qty_j": "Maximum lot size upper bound for discount tier j (or None if unbounded)",
        "C_j": "Unit purchase price for tier j ($/unit)",
        "H_j": "Annual holding cost per unit in tier j: H_j = holding_rate * C_j (or fixed H) ($/unit/year)",
        "EOQ_unconstrained_j": "Unconstrained Wilson lot size for tier j = sqrt((2 * D * S) / H_j)",
        "Q_feasible_j": "Feasible lot size satisfying tier j quantity bracket bounds",
        "APC_j": "Annual Purchase Cost = D * C_j ($/year)",
        "AOC_j": "Annual Ordering Cost = (D / Q_feasible_j) * S ($/year)",
        "AHC_j": "Annual Holding Cost = (Q_feasible_j / 2) * H_j ($/year)",
        "TAC_j": "Total Annual Cost = APC_j + AOC_j + AHC_j ($/year)",
        "N": "Replenishment order frequency = D / Q_feasible (orders/year)",
        "Total_Units_Ordered": "Total cumulative units ordered to meet demand = N * Q_feasible = D (units)",
        "Q_single": "Recommended order quantity if purchasing demand all at once = max(D, min_qty_j) (units)",
    }

    # Section 3: Formula (Symbolic)
    formula = {
        "Tier Holding Cost": "H_j = holding_cost_rate * C_j (or fixed_holding_cost)",
        "Unconstrained EOQ": "EOQ_unconstrained_j = sqrt((2 * D * S) / H_j)",
        "Feasibility Constraint": "Q_feasible_j = min_qty_j if EOQ_j < min_qty_j else (max_qty_j if (max_qty_j and EOQ_j > max_qty_j) else EOQ_j)",
        "Annual Purchase Cost": "APC_j = D * C_j",
        "Annual Ordering Cost": "AOC_j = (D / Q_feasible_j) * S",
        "Annual Holding Cost": "AHC_j = (Q_feasible_j / 2) * H_j",
        "Total Annual Cost": "TAC_j = APC_j + AOC_j + AHC_j",
        "Optimal Decision": "Optimal Tier = argmin_j(TAC_j)",
        "Demand Fulfillment Constraint": "Total Units Ordered = N * Q_batch >= D (Satisfies 100% of demand)",
        "Single Procurement Order": "Q_single = max(D, min_qty_j) >= D",
    }

    # Deterministic Intermediate Calculations across tiers
    evaluated_tiers = []
    best_tier = None
    min_total_cost = float("inf")
    substituted_formulas = {}
    steps = [
        f"1. Annual Demand (D) = {d:,.2f} units, Order Cost (S) = ${s:,.2f}",
        f"2. Holding Cost Model: {'Rate = ' + str(h_rate * 100) + '% of unit price' if h_rate is not None else 'Fixed = $' + str(fixed_h) + '/unit/year'}",
        "3. Feasible Quantity and Cost Derivation for each tier:",
    ]

    for idx, tier in enumerate(sorted_tiers, start=1):
        min_q = tier["min_qty"]
        max_q = tier["max_qty"]
        price = tier["unit_price"]

        if fixed_h is not None:
            tier_h = fixed_h
        else:
            assert h_rate is not None
            tier_h = h_rate * price

        unconstrained_eoq = math.sqrt((2.0 * d * s) / tier_h)

        if unconstrained_eoq < min_q:
            feasible_q = min_q
            status = f"EOQ ({unconstrained_eoq:.1f}) < min ({min_q:,.0f}). Adjusted to tier minimum {min_q:,.0f}."
        elif max_q is not None and unconstrained_eoq > max_q:
            feasible_q = max_q
            status = f"EOQ ({unconstrained_eoq:.1f}) > max ({max_q:,.0f}). Adjusted to tier upper limit {max_q:,.0f}."
        else:
            feasible_q = unconstrained_eoq
            status = f"EOQ ({unconstrained_eoq:.1f}) is feasible inside [{min_q:,.0f}, {max_q if max_q else 'inf'}]."

        annual_purchase = d * price
        annual_ordering = (d / feasible_q) * s
        annual_holding = (feasible_q / 2.0) * tier_h
        total_cost = annual_purchase + annual_ordering + annual_holding

        tier_summary = {
            "tier_index": idx,
            "min_qty": min_q,
            "max_qty": max_q,
            "unit_price": price,
            "holding_cost_per_unit": tier_h,
            "unconstrained_eoq": unconstrained_eoq,
            "feasible_quantity": feasible_q,
            "annual_purchase_cost": annual_purchase,
            "annual_ordering_cost": annual_ordering,
            "annual_holding_cost": annual_holding,
            "total_annual_cost": total_cost,
            "status": status,
        }
        evaluated_tiers.append(tier_summary)

        sub_desc = (
            f"Tier {idx} [{min_q:,.0f} - {max_q if max_q else 'inf'}]: "
            f"H_{idx} = ${tier_h:.2f}/unit/yr; "
            f"EOQ_{idx} = sqrt((2 * {d:,.0f} * {s:,.2f}) / {tier_h:.2f}) = {unconstrained_eoq:,.1f}; "
            f"Feasible Q_{idx} = {feasible_q:,.1f}; "
            f"TAC_{idx} = ({d:,.0f} * ${price:.2f}) + (({d:,.0f}/{feasible_q:,.1f}) * ${s:.2f}) + (({feasible_q:,.1f}/2) * ${tier_h:.2f}) = "
            f"${annual_purchase:,.2f} + ${annual_ordering:,.2f} + ${annual_holding:,.2f} = ${total_cost:,.2f}"
        )
        substituted_formulas[f"Tier_{idx}_Derivation"] = sub_desc
        steps.append(f"   • {sub_desc}")

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_tier = tier_summary

    assert best_tier is not None

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = substituted_formulas
    formula_with_substituted_values["Optimal Choice"] = (
        f"min(TAC) = Tier {best_tier['tier_index']} at Q* = {best_tier['feasible_quantity']:,.1f} units with TAC = ${best_tier['total_annual_cost']:,.2f}"
    )

    # Single order procurement quantity (must be at least demand AND at least tier minimum)
    single_order_q = max(d, best_tier["min_qty"])
    orders_per_year = d / best_tier["feasible_quantity"]

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "evaluated_tiers": evaluated_tiers,
        "best_tier_index": best_tier["tier_index"],
        "optimal_feasible_quantity": best_tier["feasible_quantity"],
        "optimal_unit_price": best_tier["unit_price"],
        "customer_demand": d,
        "recommended_order_quantity": single_order_q,
        "single_procurement_order_quantity": single_order_q,
        "annual_purchase_cost": best_tier["annual_purchase_cost"],
        "annual_ordering_cost": best_tier["annual_ordering_cost"],
        "annual_holding_cost": best_tier["annual_holding_cost"],
        "min_total_annual_cost": best_tier["total_annual_cost"],
    }

    # Section 6: Final Answer
    final_answer = {
        "optimal_tier": best_tier["tier_index"],
        "recommended_order_quantity": single_order_q,
        "recommended_order_quantity_int": int(round(single_order_q)),
        "optimal_unit_price": best_tier["unit_price"],
        "customer_demand": d,
        "single_procurement_order_quantity": single_order_q,
        "optimal_order_quantity": best_tier["feasible_quantity"],
        "min_total_annual_cost": best_tier["total_annual_cost"],
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check": "Feasibility in Selected Tier Bracket",
            "passed": best_tier["feasible_quantity"] >= best_tier["min_qty"]
            and (best_tier["max_qty"] is None or best_tier["feasible_quantity"] <= best_tier["max_qty"]),
            "details": f"Optimal Q={best_tier['feasible_quantity']:,.1f} satisfies Tier {best_tier['tier_index']} bounds [{best_tier['min_qty']:,.0f}, {best_tier['max_qty'] if best_tier['max_qty'] else 'inf'}]",
        },
        {
            "check": "Global Total Cost Minimality Across Tiers",
            "passed": all(best_tier["total_annual_cost"] <= t["total_annual_cost"] + 1e-4 for t in evaluated_tiers),
            "details": f"Tier {best_tier['tier_index']} TAC (${best_tier['total_annual_cost']:,.2f}) <= all other evaluated tiers",
        },
        {
            "check": "Demand Satisfaction Constraint (Order Meets or Exceeds Demand)",
            "passed": single_order_q >= d,
            "details": f"Recommended order quantity ({single_order_q:,.0f} units) >= demand ({d:,.0f} units). "
                       f"Order satisfies 100% of customer demand and qualifies for Tier {best_tier['tier_index']} pricing (minimum required: {best_tier['min_qty']:,.0f} units).",
        },
    ]

    # Section 8: Assumptions
    assumptions = [
        "All-units quantity discount structure (the discounted price applies to every unit in the order batch).",
        "Single order procurement satisfies at least the full customer demand in a single delivery batch.",
        "Annual demand is deterministic and known in advance.",
        "Supplier provides instantaneous fulfillment of the entire order with zero partial shipments.",
        "The purchasing firm has adequate working capital and warehouse physical storage capacity to hold the order volume.",
        "Zero deterioration, spoilage, or inventory shrinkage occurs when holding bulk order quantities.",
    ]

    if single_order_q > d:
        detail_msg = f" (exceeds demand of {round(d):,} units to qualify for the Tier {best_tier['tier_index']} discount minimum of {round(best_tier['min_qty']):,} units)"
    else:
        detail_msg = f" (fulfills 100% of the {round(d):,} unit demand and qualifies for Tier {best_tier['tier_index']})"

    recommendation = (
        f"Order {round(single_order_q):,} units under Tier {best_tier['tier_index']} at ${best_tier['unit_price']:,.2f} per unit"
        f"{detail_msg}, achieving the lowest total cost of ${best_tier['total_annual_cost']:,.2f}."
    )

    return {
        "tool_name": "calculate_quantity_discount",
        "model_name": "Quantity Discount Model",
        # The 8 Student Verification Sections:
        "inputs": inputs_extracted,
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Standard auxiliary metadata
        "step_by_step": steps,
        "results": {
            "optimal_tier": best_tier["tier_index"],
            "recommended_order_quantity": single_order_q,
            "recommended_order_quantity_int": int(round(single_order_q)),
            "recommended_order_quantity_to_meet_demand": single_order_q,
            "recommended_order_quantity_to_meet_demand_int": int(round(single_order_q)),
            "optimal_order_quantity": best_tier["feasible_quantity"],
            "single_procurement_order_quantity": single_order_q,
            "optimal_unit_price": best_tier["unit_price"],
            "min_total_annual_cost": best_tier["total_annual_cost"],
            "annual_purchase_cost": best_tier["annual_purchase_cost"],
            "annual_ordering_cost": best_tier["annual_ordering_cost"],
            "annual_holding_cost": best_tier["annual_holding_cost"],
            "all_tiers_evaluated": evaluated_tiers,
        },
        "verification": {
            "is_valid": all(c["passed"] for c in verification_checks),
            "checks": verification_checks,
        },
        "recommendation": recommendation,
    }


# ==============================================================================
# 3. Safety Stock and Reorder Point (ROP)
# ==============================================================================

def calculate_rop(
    demand_rate: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_std: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate Safety Stock (SS) and Reorder Point (ROP) under continuous review (s, Q).

    Supports 4 uncertainty scenarios:
        1. Constant demand and constant lead time (sigma_d = 0, sigma_L = 0)
        2. Variable demand and constant lead time (sigma_d > 0, sigma_L = 0)
        3. Constant demand and variable lead time (sigma_d = 0, sigma_L > 0)
        4. Variable demand and variable lead time (sigma_d > 0, sigma_L > 0):
           sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
           SS = z * sigma_LT
           ROP = d * L + SS

    Returns all 8 verification sections for student learning.
    """
    d_input = demand_rate if demand_rate is not None else kwargs.get("avg_demand")
    lt_input = lead_time if lead_time is not None else kwargs.get("lead_time")
    sd_input = demand_std if demand_std is not None else kwargs.get("std_demand", 0.0)
    slt_input = lead_time_std if lead_time_std is not None else kwargs.get("std_lead_time", 0.0)

    # 1. Input Validation (never invent missing numerical inputs)
    if d_input is None:
        raise InventoryCalculationError("Average demand 'demand_rate' is required and cannot be None.")
    if lt_input is None:
        raise InventoryCalculationError("Lead time 'lead_time' is required and cannot be None.")

    try:
        d = float(d_input)
        lt = float(lt_input)
        sd = float(sd_input) if sd_input is not None else 0.0
        slt = float(slt_input) if slt_input is not None else 0.0
    except (ValueError, TypeError) as err:
        raise InventoryCalculationError(f"Numerical inputs must be convertible to float. Error: {err}")

    if d <= 0:
        raise InventoryCalculationError(f"Average demand rate must be > 0. Received: {d}")
    if lt <= 0:
        raise InventoryCalculationError(f"Lead time must be > 0. Received: {lt}")
    if sd < 0:
        raise InventoryCalculationError(f"Demand standard deviation cannot be negative. Received: {sd}")
    if slt < 0:
        raise InventoryCalculationError(f"Lead time standard deviation cannot be negative. Received: {slt}")

    is_demand_variable = sd > 0
    is_lead_time_variable = slt > 0

    # If any variability exists, service_level or z_score is required; never invent it
    if is_demand_variable or is_lead_time_variable:
        if service_level is None and z_score is None:
            raise InventoryCalculationError(
                "Either 'service_level' or 'z_score' must be provided when demand or lead time is variable; "
                "missing service level cannot be assumed."
            )

    computed_z = 0.0
    computed_sl = 1.0
    if z_score is not None:
        computed_z = float(z_score)
        computed_sl = NormalDist().cdf(computed_z)
    elif service_level is not None:
        sl = float(service_level)
        if sl <= 0.0:
            raise InventoryCalculationError(f"Service level must be > 0. Received: {service_level}")
        if sl >= 100.0:
            raise InventoryCalculationError(f"Service level cannot be 100% or greater. Received: {service_level}")
        if sl == 1.0:
            raise InventoryCalculationError(f"Service level cannot be 1.0 (100% requires infinite inventory). Received: {service_level}")
        if sl > 1.0:
            sl = sl / 100.0
        if not (0.0 < sl < 1.0):
            raise InventoryCalculationError(f"Service level must be strictly between 0 and 1. Received: {service_level}")
        computed_sl = sl
        computed_z = NormalDist().inv_cdf(computed_sl)

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand_rate (d)": d,
        "lead_time (L)": lt,
        "demand_std (sigma_d)": sd,
        "lead_time_std (sigma_L)": slt,
        "service_level": computed_sl if (is_demand_variable or is_lead_time_variable) else None,
        "z_score": computed_z if (is_demand_variable or is_lead_time_variable) else None,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "d": "Mean demand rate per period (units/period)",
        "L": "Mean supplier delivery lead time (periods)",
        "sigma_d": "Standard deviation of periodic customer demand (units/period)",
        "sigma_L": "Standard deviation of supplier delivery lead time (periods)",
        "CSL": "Cycle Service Level / non-stockout probability per replenishment cycle",
        "z": "Standard normal distribution safety factor (z-quantile where Phi(z) = CSL)",
        "DDLT": "Expected demand during lead time = d * L (units)",
        "sigma_LT": "Standard deviation of demand during lead time (units)",
        "SS": "Safety Stock inventory buffer = z * sigma_LT (units)",
        "ROP": "Reorder Point inventory trigger level = d * L + SS (units)",
    }

    # Deterministic Intermediate Calculations across the 4 supported cases
    ddlt = d * lt

    if not is_demand_variable and not is_lead_time_variable:
        scenario_name = "Constant demand and constant lead time"
        variance_lt = 0.0
        sigma_lt = 0.0
        safety_stock = 0.0
        rop = ddlt
        formula_symbolic = {
            "Expected Lead Time Demand": "DDLT = d * L",
            "Lead Time Standard Deviation": "sigma_LT = 0 (deterministic)",
            "Safety Stock": "SS = 0",
            "Reorder Point": "ROP = d * L",
        }
        formula_substituted = {
            "DDLT": f"DDLT = {d:,.2f} * {lt:,.2f} = {ddlt:,.2f} units",
            "sigma_LT": "sigma_LT = 0.00 units",
            "Safety Stock": "SS = 0.00 units",
            "Reorder Point": f"ROP = {ddlt:,.2f} units",
        }
        assumptions = [
            "Customer demand rate is strictly deterministic and constant across all periods.",
            "Supplier delivery lead time is strictly deterministic and constant.",
            "Safety stock is zero because there is zero demand or lead time uncertainty.",
            "Continuous inventory monitoring checks stock position in real time.",
        ]
    elif is_demand_variable and not is_lead_time_variable:
        scenario_name = "Variable demand and constant lead time"
        variance_lt = lt * (sd ** 2)
        sigma_lt = math.sqrt(variance_lt)
        safety_stock = computed_z * sigma_lt
        rop = ddlt + safety_stock
        formula_symbolic = {
            "Expected Lead Time Demand": "DDLT = d * L",
            "Lead Time Standard Deviation": "sigma_LT = sqrt(L) * sigma_d",
            "Safety Stock": "SS = z * sigma_LT",
            "Reorder Point": "ROP = d * L + SS",
        }
        formula_substituted = {
            "DDLT": f"DDLT = {d:,.2f} * {lt:,.2f} = {ddlt:,.2f} units",
            "sigma_LT": f"sigma_LT = sqrt({lt:.2f}) * {sd:.2f} = {sigma_lt:,.2f} units",
            "Safety Stock": f"SS = {computed_z:.4f} * {sigma_lt:,.2f} = {safety_stock:,.2f} units",
            "Reorder Point": f"ROP = {ddlt:,.2f} + {safety_stock:,.2f} = {rop:,.2f} units",
        }
        assumptions = [
            "Customer demand per period is an independent, identically distributed normal random variable.",
            "Supplier delivery lead time is completely constant and deterministic.",
            "Stockout risk is evaluated over the cycle lead-time window.",
            "Unfilled demand during replenishment is backordered.",
        ]
    elif not is_demand_variable and is_lead_time_variable:
        scenario_name = "Constant demand and variable lead time"
        variance_lt = (d ** 2) * (slt ** 2)
        sigma_lt = math.sqrt(variance_lt)
        safety_stock = computed_z * sigma_lt
        rop = ddlt + safety_stock
        formula_symbolic = {
            "Expected Lead Time Demand": "DDLT = d * L",
            "Lead Time Standard Deviation": "sigma_LT = d * sigma_L",
            "Safety Stock": "SS = z * sigma_LT",
            "Reorder Point": "ROP = d * L + SS",
        }
        formula_substituted = {
            "DDLT": f"DDLT = {d:,.2f} * {lt:,.2f} = {ddlt:,.2f} units",
            "sigma_LT": f"sigma_LT = {d:.2f} * {slt:.2f} = {sigma_lt:,.2f} units",
            "Safety Stock": f"SS = {computed_z:.4f} * {sigma_lt:,.2f} = {safety_stock:,.2f} units",
            "Reorder Point": f"ROP = {ddlt:,.2f} + {safety_stock:,.2f} = {rop:,.2f} units",
        }
        assumptions = [
            "Demand rate is constant and deterministic.",
            "Supplier delivery lead time is a normal random variable with known mean and variance.",
            "Safety stock buffer protects strictly against supplier delivery delays.",
        ]
    else:
        # Case 4: Variable Demand and Variable Lead Time
        scenario_name = "Variable demand and variable lead time"
        term1 = lt * (sd ** 2)
        term2 = (d ** 2) * (slt ** 2)
        variance_lt = term1 + term2
        sigma_lt = math.sqrt(variance_lt)
        safety_stock = computed_z * sigma_lt
        rop = ddlt + safety_stock
        formula_symbolic = {
            "Expected Lead Time Demand": "DDLT = d * L",
            "Lead Time Standard Deviation": "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)",
            "Safety Stock": "SS = z * sigma_LT",
            "Reorder Point": "ROP = d * L + SS",
        }
        formula_substituted = {
            "DDLT": f"DDLT = {d:,.2f} * {lt:,.2f} = {ddlt:,.2f} units",
            "sigma_LT": (
                f"sigma_LT = sqrt({lt:.2f} * {sd:.2f}^2 + {d:.2f}^2 * {slt:.2f}^2) = "
                f"sqrt({term1:,.2f} + {term2:,.2f}) = sqrt({variance_lt:,.2f}) = {sigma_lt:,.2f} units"
            ),
            "Safety Stock": f"SS = {computed_z:.4f} * {sigma_lt:,.2f} = {safety_stock:,.2f} units",
            "Reorder Point": f"ROP = {ddlt:,.2f} + {safety_stock:,.2f} = {rop:,.2f} units",
        }
        assumptions = [
            "Periodic demand and delivery lead times are independent normal random variables.",
            "Full bivariate variance convolution formula accurately characterizes lead-time exposure.",
            "Continuous review (s, Q) monitors stock position to trigger orders exactly when ROP is hit.",
            "Cycle Service Level CSL measures the probability of zero stockouts per replenishment cycle.",
        ]

    stockout_risk = 1.0 - computed_sl

    # Section 3: Formula
    formula = formula_symbolic

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = formula_substituted

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "scenario": scenario_name,
        "uncertainty_scenario": scenario_name,
        "expected_demand_during_lead_time": ddlt,
        "variance_during_lead_time": variance_lt,
        "sigma_LT": sigma_lt,
        "std_dev_demand_during_lead_time": sigma_lt,
        "z_score": computed_z,
        "service_level": computed_sl,
        "stockout_risk_per_cycle": stockout_risk,
        "safety_stock": safety_stock,
        "reorder_point": rop,
    }

    # Section 6: Final Answer
    final_answer = {
        "reorder_point": rop,
        "recommended_reorder_point_int": int(math.ceil(rop)),
        "safety_stock": safety_stock,
        "recommended_safety_stock_int": int(math.ceil(safety_stock)),
        "expected_lead_time_demand": ddlt,
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check": "Non-negative Safety Stock for CSL >= 50%",
            "passed": safety_stock >= 0 if computed_sl >= 0.50 else True,
            "details": f"SS={safety_stock:.2f} units at CSL={computed_sl * 100:.2f}%",
        },
        {
            "check": "Reorder Point Exceeds Expected Lead Time Demand",
            "passed": rop >= ddlt if computed_sl >= 0.50 else True,
            "details": f"ROP ({rop:.2f}) >= DDLT ({ddlt:.2f})",
        },
        {
            "check": "Non-negative Standard Deviation Invariant (sigma_LT >= 0)",
            "passed": sigma_lt >= 0,
            "details": f"sigma_LT={sigma_lt:.2f}",
        },
    ]

    # Section 8: Assumptions
    # Assigned in scenario branch above

    steps = [
        f"1. Uncertainty Scenario: {scenario_name}",
        f"2. Average Demand per period (d) = {d:,.2f}",
        f"3. Mean Lead Time (L) = {lt:,.2f} periods",
        f"4. Expected Demand During Lead Time (DDLT) = d * L = {d:,.2f} * {lt:,.2f} = {ddlt:,.2f} units",
    ]
    if is_demand_variable or is_lead_time_variable:
        steps.append(f"5. Target Service Level = {computed_sl * 100:.2f}% -> Standard Normal Z = {computed_z:.4f}")
        steps.append(f"6. Lead Time Std Dev: {formula_substituted.get('sigma_LT')}")
        steps.append(f"7. Safety Stock (SS) = z * sigma_LT = {computed_z:.4f} * {sigma_lt:.2f} = {safety_stock:,.2f} units")
        steps.append(f"8. Reorder Point (ROP) = d * L + SS = {ddlt:,.2f} + {safety_stock:,.2f} = {rop:,.2f} units")
    else:
        steps.append("5. Zero uncertainty detected -> Safety Stock (SS) = 0.00 units")
        steps.append(f"6. Reorder Point (ROP) = d * L = {rop:,.2f} units")

    recommendation = (
        f"Set Reorder Point (ROP) to {int(math.ceil(rop)):,} units (exact: {rop:.1f}). "
        f"Place an order whenever on-hand stock drops to or below this threshold. "
        f"This policy holds {int(math.ceil(safety_stock)):,} units of safety stock under the '{scenario_name}' model."
    )

    return {
        "tool_name": "calculate_rop",
        "model_name": "Safety Stock & Reorder Point (Continuous Review)",
        # The 8 Student Verification Sections:
        "inputs": inputs_extracted,
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Standard auxiliary metadata
        "step_by_step": steps,
        "results": {
            "reorder_point": rop,
            "recommended_reorder_point_int": int(math.ceil(rop)),
            "safety_stock": safety_stock,
            "recommended_safety_stock_int": int(math.ceil(safety_stock)),
            **intermediate_calculations,
        },
        "verification": {
            "is_valid": all(c["passed"] for c in verification_checks),
            "checks": verification_checks,
        },
        "recommendation": recommendation,
    }


# Backwards-compatible alias
calculate_safety_stock_rop = calculate_rop


# ==============================================================================
# 4. Fixed-Period Review (Periodic Review / Order-Up-To Policy)
# ==============================================================================

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_rate: Optional[float] = None,
    demand_std: Optional[float] = None,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    current_inventory: Optional[float] = None,
    inventory_position: Optional[float] = None,
    safety_stock: Optional[float] = None,
    on_order: float = 0.0,
    backorders: float = 0.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate target inventory (order-up-to level) and replenishment order for periodic review (R, S).

    Formulations:
        Protection period = review period + lead time
        Target inventory (M) = demand during protection period + safety stock
        Order quantity (Q) = max(0, target inventory - inventory position)

    Returns all 8 verification sections for student learning.
    """
    r_input = review_period if review_period is not None else kwargs.get("review_period")
    lt_input = lead_time if lead_time is not None else kwargs.get("lead_time")
    d_input = demand_rate if demand_rate is not None else kwargs.get("avg_demand")
    sd_input = demand_std if demand_std is not None else kwargs.get("std_demand")
    cur_inv = current_inventory if current_inventory is not None else kwargs.get("current_inventory")
    inv_pos = inventory_position if inventory_position is not None else kwargs.get("inventory_position")
    explicit_ss = safety_stock if safety_stock is not None else kwargs.get("safety_stock")
    oo_input = on_order if on_order is not None else kwargs.get("on_order", 0.0)
    bo_input = backorders if backorders is not None else kwargs.get("backorders", 0.0)

    # 1. Input Validation (never invent missing numerical inputs)
    if r_input is None:
        raise InventoryCalculationError("Review period 'review_period' is required and cannot be None.")
    if lt_input is None:
        raise InventoryCalculationError("Lead time 'lead_time' is required and cannot be None.")
    if d_input is None:
        raise InventoryCalculationError("Demand rate 'demand_rate' is required and cannot be None.")
    if explicit_ss is None:
        if sd_input is None:
            raise InventoryCalculationError("Demand standard deviation 'demand_std' is required and cannot be None.")
        if service_level is None and z_score is None:
            raise InventoryCalculationError(
                "Either 'service_level' or 'z_score' must be provided; missing service level cannot be assumed."
            )
    if cur_inv is None and inv_pos is None:
        raise InventoryCalculationError(
            "Must provide either 'current_inventory' or 'inventory_position'; missing inventory status cannot be assumed."
        )

    try:
        r = float(r_input)
        lt = float(lt_input)
        d = float(d_input)
        sd = float(sd_input) if sd_input is not None else 0.0
        oo = float(oo_input) if oo_input is not None else 0.0
        bo = float(bo_input) if bo_input is not None else 0.0
    except (ValueError, TypeError) as err:
        raise InventoryCalculationError(f"Numerical inputs must be convertible to float. Error: {err}")

    if r <= 0:
        raise InventoryCalculationError(f"Review period must be > 0. Received: {r}")
    if lt < 0:
        raise InventoryCalculationError(f"Lead time must be >= 0. Received: {lt}")
    if d <= 0:
        raise InventoryCalculationError(f"Demand rate must be > 0. Received: {d}")
    if sd < 0:
        raise InventoryCalculationError(f"Demand standard deviation cannot be negative. Received: {sd}")

    if explicit_ss is not None:
        computed_ss = float(explicit_ss)
        computed_z = 0.0
        computed_sl = 1.0
    elif z_score is not None:
        computed_z = float(z_score)
        computed_sl = NormalDist().cdf(computed_z)
        computed_ss = None
    else:
        assert service_level is not None
        sl = float(service_level)
        if sl <= 0.0 or sl >= 100.0 or sl == 1.0:
            raise InventoryCalculationError(f"Service level must be strictly between 0 and 1. Received: {service_level}")
        if sl > 1.0:
            sl = sl / 100.0
        computed_sl = sl
        computed_z = NormalDist().inv_cdf(computed_sl)
        computed_ss = None

    if inv_pos is not None:
        effective_ip = float(inv_pos)
    else:
        assert cur_inv is not None
        effective_ip = float(cur_inv) + oo - bo

    # Deterministic Intermediate Calculations
    protection_period = r + lt
    demand_during_protection = d * protection_period
    sigma_protection = math.sqrt(protection_period) * sd
    safety_stock = computed_ss if computed_ss is not None else computed_z * sigma_protection
    target_inventory = demand_during_protection + safety_stock
    order_quantity = max(0.0, target_inventory - effective_ip)

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "review_period (R)": r,
        "lead_time (L)": lt,
        "demand_rate (d)": d,
        "demand_std (sigma_d)": sd,
        "service_level": computed_sl,
        "z_score": computed_z,
        "current_inventory": float(cur_inv) if cur_inv is not None else None,
        "inventory_position": effective_ip,
        "on_order": oo,
        "backorders": bo,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "R": "Review period duration between consecutive inventory reviews (periods)",
        "L": "Supplier delivery lead time (periods)",
        "P": "Protection period / total exposure window = R + L (periods)",
        "d": "Mean customer demand per period (units/period)",
        "sigma_d": "Standard deviation of periodic customer demand (units/period)",
        "CSL": "Cycle service level / probability of zero stockouts during protection period",
        "z": "Standard normal distribution safety factor (quantile where Phi(z) = CSL)",
        "DD_P": "Expected demand during protection period = d * (R + L) (units)",
        "sigma_P": "Standard deviation of demand over protection period = sqrt(R + L) * sigma_d (units)",
        "SS": "Safety Stock inventory buffer = z * sigma_P (units)",
        "M": "Target inventory / Order-up-to level = DD_P + SS (units)",
        "IP": "Effective inventory position = On_Hand + On_Order - Backorders (units)",
        "Q": "Recommended order batch quantity = max(0, M - IP) (units)",
    }

    # Section 3: Formula (Symbolic)
    formula = {
        "Protection Period": "P = R + L",
        "Demand During Protection Period": "DD_P = d * (R + L)",
        "Sigma Protection Period": "sigma_P = sqrt(R + L) * sigma_d",
        "Safety Stock": "SS = z * sigma_P = z * sqrt(R + L) * sigma_d",
        "Target Inventory Level": "M = DD_P + SS = d * (R + L) + z * sqrt(R + L) * sigma_d",
        "Inventory Position": "IP = On_Hand + On_Order - Backorders",
        "Order Quantity": "Q = max(0, M - IP)",
    }

    # Deterministic Intermediate Calculations
    protection_period = r + lt
    demand_during_protection = d * protection_period
    sigma_protection = math.sqrt(protection_period) * sd
    safety_stock = computed_ss if computed_ss is not None else computed_z * sigma_protection
    target_inventory = demand_during_protection + safety_stock
    order_quantity = max(0.0, target_inventory - effective_ip)

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = {
        "Protection Period Substitution": f"P = {r:,.2f} + {lt:,.2f} = {protection_period:,.2f} periods",
        "Demand During Protection Period Substitution": f"DD_P = {d:,.2f} * {protection_period:,.2f} = {demand_during_protection:,.2f} units",
        "Sigma Protection Period Substitution": f"sigma_P = sqrt({protection_period:,.2f}) * {sd:,.2f} = {sigma_protection:,.2f} units",
        "Safety Stock Substitution": f"SS = {computed_z:.4f} * {sigma_protection:,.2f} = {safety_stock:,.2f} units",
        "Target Inventory Level Substitution": f"M = {demand_during_protection:,.2f} + {safety_stock:,.2f} = {target_inventory:,.2f} units",
        "Inventory Position Substitution": f"IP = {effective_ip:,.2f} units",
        "Order Quantity Substitution": f"Q = max(0, {target_inventory:,.2f} - {effective_ip:,.2f}) = {order_quantity:,.2f} units",
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "protection_period": protection_period,
        "protection_interval": protection_period,
        "demand_during_protection_period": demand_during_protection,
        "expected_demand_protection_window": demand_during_protection,
        "sigma_protection_period": sigma_protection,
        "std_dev_protection_window": sigma_protection,
        "safety_stock": safety_stock,
        "target_inventory": target_inventory,
        "target_order_up_to_level": target_inventory,
        "inventory_position": effective_ip,
        "z_score": computed_z,
        "service_level": computed_sl,
        "order_quantity": order_quantity,
    }

    # Section 6: Final Answer
    final_answer = {
        "recommended_order_quantity": order_quantity,
        "recommended_order_quantity_int": int(math.ceil(order_quantity)),
        "target_inventory": target_inventory,
        "target_level_int": int(math.ceil(target_inventory)),
        "protection_period": protection_period,
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check": "Target Inventory Exceeds Mean Protection Period Demand",
            "passed": target_inventory >= demand_during_protection if computed_sl >= 0.50 else True,
            "details": f"Target Level M ({target_inventory:.2f}) >= Protection Demand ({demand_during_protection:.2f})",
        },
        {
            "check": "Non-negative Replenishment Order Quantity (Q >= 0)",
            "passed": order_quantity >= 0,
            "details": f"Recommended order Q={order_quantity:.2f} units",
        },
        {
            "check": "Positive Protection Exposure Window (P = R + L > 0)",
            "passed": protection_period > 0,
            "details": f"Protection period (R + L) = {r:.2f} + {lt:.2f} = {protection_period:.2f} periods",
        },
    ]

    # Section 8: Assumptions
    assumptions = [
        "Inventory audits and orders occur strictly at discrete periodic review intervals of length R.",
        "Replenishment batches arrive after fixed delivery lead time L, leaving the system exposed across (R + L).",
        "Periodic demand is normally distributed, stationary, and mutually independent across cycles.",
        "Linear inventory accounting accurately reflects effective inventory position: IP = On_Hand + On_Order - Backorders.",
        "Unfulfilled customer demand during stockout periods is backordered and completely fulfilled upon replenishment arrival.",
    ]

    steps = [
        f"1. Review Period (R) = {r:,.2f} periods, Lead Time (L) = {lt:,.2f} periods",
        f"2. Protection Period = Review Period + Lead Time = {r:,.2f} + {lt:,.2f} = {protection_period:,.2f} periods",
        f"3. Demand Rate (d) = {d:,.2f} units/period",
        f"4. Demand During Protection Period = d * (R + L) = {d:,.2f} * {protection_period:,.2f} = {demand_during_protection:,.2f} units",
        f"5. Sigma Protection Period = sqrt(R + L) * sigma_d = sqrt({protection_period:,.2f}) * {sd:,.2f} = {sigma_protection:,.2f} units",
        f"6. Target Service Level = {computed_sl * 100:.2f}% -> Standard Normal Z = {computed_z:.4f}",
        f"7. Safety Stock = Z * Sigma_Protection = {computed_z:.4f} * {sigma_protection:,.2f} = {safety_stock:,.2f} units",
        f"8. Target Inventory (Order-Up-To Level M) = Protection Demand + Safety Stock = {demand_during_protection:,.2f} + {safety_stock:,.2f} = {target_inventory:,.2f} units",
        f"9. Effective Inventory Position (IP) = {effective_ip:,.2f} units",
        f"10. Recommended Order Quantity = max(0, Target Inventory - IP) = max(0, {target_inventory:,.2f} - {effective_ip:,.2f}) = {order_quantity:,.2f} units",
    ]

    if order_quantity == 0:
        recommendation = (
            f"Do not place an order. Current inventory position ({effective_ip:,.1f} units) "
            f"already meets or exceeds the target inventory level of {target_inventory:,.1f} units."
        )
    else:
        recommendation = (
            f"Order {int(math.ceil(order_quantity)):,} units (exact: {order_quantity:.1f}) to restore "
            f"inventory position to target level of {target_inventory:,.1f} units over the {protection_period:.1f}-period protection window."
        )

    return {
        "tool_name": "calculate_periodic_review",
        "model_name": "Fixed-Period Review (Order-Up-To Policy)",
        # The 8 Student Verification Sections:
        "inputs": inputs_extracted,
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Standard auxiliary metadata
        "step_by_step": steps,
        "results": {
            "recommended_order_quantity": order_quantity,
            "recommended_order_quantity_int": int(math.ceil(order_quantity)),
            "target_inventory": target_inventory,
            "target_order_up_to_level": target_inventory,
            "target_level_int": int(math.ceil(target_inventory)),
            **intermediate_calculations,
        },
        "verification": {
            "is_valid": all(c["passed"] for c in verification_checks),
            "checks": verification_checks,
        },
        "recommendation": recommendation,
    }


# Backwards-compatible alias
calculate_fixed_period_review = calculate_periodic_review


# ==============================================================================
# 5. Newsvendor Model (Single-Period Probabilistic)
# ==============================================================================

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    shortage_penalty: float = 0.0,
    penalty: Optional[float] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity for the Newsvendor model.

    Formulations:
        Underage cost (Cu) = selling_price - cost + shortage_penalty  (or directly specified)
        Overage cost (Co)  = cost - salvage_value                      (or directly specified)
        Critical ratio (CR) = Cu / (Cu + Co)
        Recommended order quantity (Q*) = mean_demand + z * std_demand, where Phi(z) = CR

    Returns all 8 verification sections for student learning.
    """
    g_input = penalty if penalty is not None else shortage_penalty
    cu_input = underage_cost if underage_cost is not None else kwargs.get("cu")
    co_input = overage_cost if overage_cost is not None else kwargs.get("co")
    mu_input = mean_demand if mean_demand is not None else kwargs.get("mean_demand")
    sigma_input = std_demand if std_demand is not None else kwargs.get("std_demand")

    # 1. Input Validation (never invent missing numerical inputs)
    if mu_input is None:
        raise InventoryCalculationError("Mean demand 'mean_demand' is required and cannot be None.")
    if sigma_input is None:
        raise InventoryCalculationError("Demand standard deviation 'std_demand' is required and cannot be None.")

    # Determine whether underage and overage are given directly or via price/cost/salvage
    has_direct_cu_co = (cu_input is not None) and (co_input is not None)
    has_price_cost_salvage = (selling_price is not None) and (cost is not None) and (salvage_value is not None)

    if not has_direct_cu_co and not has_price_cost_salvage:
        if selling_price is None and cu_input is None:
            raise InventoryCalculationError(
                "Must provide either ('selling_price', 'cost', 'salvage_value') or ('underage_cost', 'overage_cost'); "
                "missing unit economic inputs cannot be assumed."
            )
        if cost is None and co_input is None:
            raise InventoryCalculationError("Unit cost 'cost' (or 'overage_cost') is required and cannot be None.")
        if salvage_value is None and co_input is None:
            raise InventoryCalculationError("Salvage value 'salvage_value' (or 'overage_cost') is required and cannot be None.")

    try:
        mu = float(mu_input)
        sigma = float(sigma_input)
        g = float(g_input) if g_input is not None else 0.0
        p = float(selling_price) if selling_price is not None else None
        c = float(cost) if cost is not None else None
        s = float(salvage_value) if salvage_value is not None else None
        direct_cu = float(cu_input) if cu_input is not None else None
        direct_co = float(co_input) if co_input is not None else None
    except (ValueError, TypeError) as err:
        raise InventoryCalculationError(f"Numerical inputs must be convertible to float. Error: {err}")

    if mu <= 0:
        raise InventoryCalculationError(f"Mean demand must be > 0. Received: {mu}")
    if sigma <= 0:
        raise InventoryCalculationError(f"Demand standard deviation must be > 0. Received: {sigma}")
    if g < 0:
        raise InventoryCalculationError(f"Shortage penalty cannot be negative. Received: {g}")

    if has_direct_cu_co:
        assert direct_cu is not None and direct_co is not None
        if direct_cu <= 0:
            raise InventoryCalculationError(f"Underage cost Cu must be > 0. Received: {direct_cu}")
        if direct_co <= 0:
            raise InventoryCalculationError(f"Overage cost Co must be > 0. Received: {direct_co}")
        cu = direct_cu
        co = direct_co
    else:
        assert p is not None and c is not None and s is not None
        if p <= c:
            raise InventoryCalculationError(
                f"Selling price (${p:.2f}) must exceed unit cost (${c:.2f}) for positive operating margin."
            )
        if c <= s:
            raise InventoryCalculationError(
                f"Unit cost (${c:.2f}) must exceed salvage value (${s:.2f}) to have an overage loss risk."
            )
        cu = p - c + g
        co = c - s

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "selling_price (p)": p,
        "cost (c)": c,
        "salvage_value (s)": s,
        "underage_cost (Cu)": cu,
        "overage_cost (Co)": co,
        "mean_demand (mu)": mu,
        "std_demand (sigma)": sigma,
        "shortage_penalty (g)": g,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "p": "Unit selling price / sales revenue ($/unit)",
        "c": "Unit procurement or production cost ($/unit)",
        "s": "Unit salvage value / clearance revenue on unsold stock ($/unit)",
        "g": "Unit shortage penalty / customer goodwill loss ($/unit)",
        "mu": "Mean customer demand for the single selling period (units)",
        "sigma": "Standard deviation of demand for the single selling period (units)",
        "Cu": "Cost of underage (lost net profit per unfulfilled unit of demand) = p - c + g ($/unit)",
        "Co": "Cost of overage (net loss per unsold unit left over at end of period) = c - s ($/unit)",
        "CR": "Critical ratio / optimal service fractile = Cu / (Cu + Co)",
        "z": "Standard normal distribution safety factor satisfying Phi(z) = CR",
        "Q*": "Recommended optimal order quantity = mu + z * sigma (units)",
    }

    # Section 3: Formula (Symbolic)
    formula = {
        "Underage Cost": "Cu = p - c + g (or direct underage_cost)",
        "Overage Cost": "Co = c - s (or direct overage_cost)",
        "Critical Ratio": "CR = Cu / (Cu + Co)",
        "Standard Normal Quantile": "z = Phi^-1(CR)",
        "Optimal Order Quantity": "Q* = mu + z * sigma",
        "Expected Lost Sales": "E[Lost Sales] = sigma * [phi(z) - z * (1 - CR)]",
        "Expected Sales": "E[Sales] = mu - E[Lost Sales]",
        "Expected Leftover Inventory": "E[Leftover] = Q* - E[Sales]",
        "Expected Profit": "E[Profit] = (p * E[Sales]) + (s * E[Leftover]) - (c * Q*) - (g * E[Lost Sales])",
    }

    # Deterministic Intermediate Calculations
    critical_ratio = cu / (cu + co)

    dist = NormalDist(mu=0.0, sigma=1.0)
    z = dist.inv_cdf(critical_ratio)
    optimal_q = mu + z * sigma

    phi_z = dist.pdf(z)
    loss_function_z = phi_z - z * (1.0 - critical_ratio)
    expected_lost_sales = sigma * loss_function_z
    expected_sales = mu - expected_lost_sales
    expected_leftover = optimal_q - expected_sales

    expected_revenue = (p * expected_sales) + (s * expected_leftover)
    expected_purchase_cost = c * optimal_q
    expected_penalty_cost = g * expected_lost_sales
    expected_profit = expected_revenue - expected_purchase_cost - expected_penalty_cost

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = {
        "Underage Cost Substitution": f"Cu = ${p:,.2f} - ${c:,.2f}" + (f" + ${g:,.2f}" if g > 0 else "") + f" = ${cu:,.2f}/unit",
        "Overage Cost Substitution": f"Co = ${c:,.2f} - ${s:,.2f} = ${co:,.2f}/unit",
        "Critical Ratio Substitution": f"CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {critical_ratio:.4f} ({critical_ratio * 100:.2f}%)",
        "Z-Quantile Substitution": f"z = Phi^-1({critical_ratio:.4f}) = {z:.4f}",
        "Optimal Order Quantity Substitution": f"Q* = {mu:,.2f} + ({z:.4f} * {sigma:,.2f}) = {optimal_q:,.2f} units",
        "Expected Profit Substitution": f"E[Profit] = ${expected_revenue:,.2f} - ${expected_purchase_cost:,.2f}" + (f" - ${expected_penalty_cost:,.2f}" if g > 0 else "") + f" = ${expected_profit:,.2f}",
    }

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "underage_cost": cu,
        "cost_of_underage": cu,
        "overage_cost": co,
        "cost_of_overage": co,
        "critical_ratio": critical_ratio,
        "z_score": z,
        "optimal_order_quantity": optimal_q,
        "expected_lost_sales": expected_lost_sales,
        "expected_sales": expected_sales,
        "expected_leftover_inventory": expected_leftover,
        "expected_revenue": expected_revenue,
        "expected_purchase_cost": expected_purchase_cost,
        "expected_penalty_cost": expected_penalty_cost,
        "expected_profit": expected_profit,
    }

    # Section 6: Final Answer
    final_answer = {
        "underage_cost": cu,
        "overage_cost": co,
        "critical_ratio": critical_ratio,
        "recommended_order_quantity": optimal_q,
        "recommended_order_quantity_int": int(round(optimal_q)),
        "expected_profit": expected_profit,
    }

    # Section 7: Verification Checks
    verification_checks = [
        {
            "check": "Economic Viability Constraint (Cu > 0 and Co > 0)",
            "passed": cu > 0 and co > 0,
            "details": f"Underage Cost Cu=${cu:.2f}, Overage Cost Co=${co:.2f}",
        },
        {
            "check": "Valid Critical Fractile Probability (0 < CR < 1)",
            "passed": 0.0 < critical_ratio < 1.0,
            "details": f"CR = {critical_ratio:.4f} ({critical_ratio * 100:.2f}%)",
        },
        {
            "check": "Order Direction Consistency with Critical Fractile",
            "passed": (optimal_q >= mu if critical_ratio >= 0.50 else optimal_q < mu),
            "details": f"CR={critical_ratio:.2f} -> Q* ({optimal_q:.1f}) {' >= ' if critical_ratio >= 0.50 else ' < '} Mean ({mu:,.0f})",
        },
        {
            "check": "Positive Expected Operating Profit",
            "passed": expected_profit > 0,
            "details": f"Expected Profit = ${expected_profit:,.2f} > 0",
        },
    ]

    # Section 8: Assumptions
    assumptions = [
        "Single sales period with perishable, seasonal, or fashion merchandise.",
        "Zero mid-season replenishment; entire inventory batch must be committed before demand realizes.",
        "Customer demand is normally distributed with known stationary mean (mu) and standard deviation (sigma).",
        "All unsold inventory at the end of the season is liquidated at a constant salvage recovery price s.",
        "Underage cost accurately reflects lost gross margin plus any explicit customer goodwill loss penalty.",
    ]

    steps = [
        f"1. Selling Price (p) = ${p:,.2f}, Unit Cost (c) = ${c:,.2f}, Salvage Value (s) = ${s:,.2f}" + (f", Penalty (g) = ${g:,.2f}" if g > 0 else ""),
        f"2. Mean Demand (mu) = {mu:,.2f}, Demand Std Dev (sigma) = {sigma:,.2f}",
        f"3. Underage Cost: Cu = p - c + g = ${p:,.2f} - ${c:,.2f}" + (f" + ${g:,.2f}" if g > 0 else "") + f" = ${cu:,.2f}",
        f"4. Overage Cost: Co = c - s = ${c:,.2f} - ${s:,.2f} = ${co:,.2f}",
        f"5. Critical Ratio: CR = Cu / (Cu + Co) = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = {critical_ratio:.4f} ({critical_ratio * 100:.2f}%)",
        f"6. Standard Normal Z = Phi^-1({critical_ratio:.4f}) = {z:.4f}",
        f"7. Recommended Order Quantity Q* = mu + z * sigma = {mu:,.2f} + ({z:.4f} * {sigma:,.2f}) = {optimal_q:,.2f} units",
        f"8. Expected Lost Sales = {expected_lost_sales:,.2f} units",
        f"9. Expected Sales = {expected_sales:,.2f} units",
        f"10. Expected Leftover Inventory = {expected_leftover:,.2f} units",
        f"11. Expected Profit = ${expected_profit:,.2f}",
    ]

    recommendation = (
        f"Order {int(round(optimal_q)):,} units (exact: {optimal_q:.1f} units). "
        f"Because the critical ratio is {critical_ratio * 100:.1f}%, stocking above the mean "
        f"optimizes profitability by balancing underage margin (${cu:.2f}/unit) against overage risk (${co:.2f}/unit). "
        f"Projected expected profit is ${expected_profit:,.2f}."
    )

    return {
        "tool_name": "calculate_newsvendor",
        "model_name": "Newsvendor Model (Single-Period Probabilistic)",
        # The 8 Student Verification Sections:
        "inputs": inputs_extracted,
        "inputs_extracted": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formula,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Standard auxiliary metadata
        "step_by_step": steps,
        "results": {
            "optimal_order_quantity": optimal_q,
            "recommended_order_quantity_int": int(round(optimal_q)),
            **intermediate_calculations,
        },
        "verification": {
            "is_valid": all(c["passed"] for c in verification_checks),
            "checks": verification_checks,
        },
        "recommendation": recommendation,
    }


# ==============================================================================
# 6. Multi-Criteria Inventory Classification (Sales & Hits)
# ==============================================================================

def classify_inventory_multicriteria(
    df: Any,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    item_num_col: str = "Item Number",
    item_name_col: str = "Item Name",
    uom_col: str = "UoM",
    thresholds: Optional[Dict[str, float]] = None,
    sales_weight: float = 0.65,
    hits_weight: float = 0.35,
) -> Dict[str, Any]:
    """Perform multi-criteria inventory classification using Sales and POS Hits.

    Implementation Steps:
    - Step 1: Classify Sales and POS Hits separately using Pareto cumulative thresholds:
              0-60%: A, 61-80%: B, 81-90%: C, 91-100%: D.
              Convert tiers to scores: A = 4, B = 3, C = 2, D = 1.
    - Step 2: Combined Weighted Score = 0.65*(Sales Score) + 0.35*(Hits Score).
    - Step 3: Assign Final Inventory Class:
              A: 4.00 - 3.25
              B: 3.24 - 2.50
              C: 2.49 - 1.75
              D: < 1.75
    - Step 4: Classification summary reporting item count and total Sales per category.

    Parameters:
        df: pandas DataFrame containing inventory item records.
        sales_col: Name of column containing Sales ($) values. Auto-detected if None.
        hits_col: Name of column containing POS Hits counts. Auto-detected if None.
        item_num_col: Name of column containing Item Number/SKU.
        item_name_col: Name of column containing Item Name.
        uom_col: Name of column containing Unit of Measure.
        thresholds: Cumulative cutoff thresholds for classes A, B, C, D.
                    Defaults to {"A": 0.60, "B": 0.80, "C": 0.90, "D": 1.00}.
        sales_weight: Weight for Sales in composite score (default: 0.65).
        hits_weight: Weight for POS Hits in composite score (default: 0.35).

    Returns:
        Structured dictionary containing:
        - classified_df: DataFrame with appended classification columns.
        - class_summary: DataFrame summarizing Item Count and Total Sales by category.
        - matrix_crosstab: 2D cross-tabulation table of Sales Class vs Hits Class.
        - total_skus: Total number of inventory items.
        - total_sales: Total revenue across all items.
        - total_hits: Total POS Hits across all items.
        - class_counts: Dictionary mapping A, B, C, D to item counts.
        - sales_col: Name of resolved sales column.
        - hits_col: Name of resolved hits column.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise InventoryCalculationError("Input data must be a pandas DataFrame.")

    if df.empty:
        raise InventoryCalculationError("Input DataFrame is empty; cannot perform inventory classification.")

    # Auto-detect Sales column if not explicitly provided
    if sales_col is None or sales_col not in df.columns:
        for candidate in ["Sales ($)", "Sales", "sales", "SALES", "Sales_Dollar", "Revenue"]:
            if candidate in df.columns:
                sales_col = candidate
                break

    if sales_col is None or sales_col not in df.columns:
        raise InventoryCalculationError(
            f"Missing required Sales column in DataFrame. Available columns: {list(df.columns)}"
        )

    # Auto-detect Hits / POS Hits column if not explicitly provided
    if hits_col is None or hits_col not in df.columns:
        for candidate in ["POS Hits", "pos hits", "POS_Hits", "Hits", "hits", "HITS", "Hit", "Frequency", "Picks"]:
            if candidate in df.columns:
                hits_col = candidate
                break

    if hits_col is None or hits_col not in df.columns:
        raise InventoryCalculationError(
            f"Missing required POS Hits column in DataFrame. Available columns: {list(df.columns)}"
        )

    # Copy DataFrame to avoid modifying original input
    res_df = df.copy()

    # Clean numeric columns (handling dollar signs, commas, and string formatting)
    res_df["_sales_clean"] = pd.to_numeric(
        res_df[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0.0)

    res_df["_hits_clean"] = pd.to_numeric(
        res_df[hits_col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0.0)

    # Pareto Thresholds: 0-60%: A, 61-80%: B, 81-90%: C, 91-100%: D
    if thresholds is None:
        thresholds = {"A": 0.60, "B": 0.80, "C": 0.90, "D": 1.00}

    def assign_tier(cum_pct: float, val: float, rank_idx: int) -> str:
        if val <= 0:
            return "D"
        if rank_idx == 0 or cum_pct <= thresholds["A"]:
            return "A"
        elif cum_pct <= thresholds["B"]:
            return "B"
        elif cum_pct <= thresholds["C"]:
            return "C"
        else:
            return "D"

    # Step 1a: Sales Classification (Pareto Cumulative Distribution: 0-60% A, 61-80% B, 81-90% C, 91-100% D)
    sales_sorted = res_df.sort_values(by="_sales_clean", ascending=False).reset_index().copy()
    tot_sales = sales_sorted["_sales_clean"].sum()
    cum_sales = sales_sorted["_sales_clean"].cumsum()
    div_sales = tot_sales if tot_sales > 0 else 1.0
    sales_sorted["_cum_pct"] = cum_sales / div_sales
    sales_sorted["Sales Cumulative %"] = (sales_sorted["_cum_pct"] * 100.0).round(2)

    sales_sorted["Sales Class"] = [
        assign_tier(p, v, i) for i, (p, v) in enumerate(zip(sales_sorted["_cum_pct"], sales_sorted["_sales_clean"]))
    ]

    # Step 1b: POS Hits Classification (Velocity Cumulative Distribution: 0-60% A, 61-80% B, 81-90% C, 91-100% D)
    hits_sorted = res_df.sort_values(by="_hits_clean", ascending=False).reset_index().copy()
    tot_hits = hits_sorted["_hits_clean"].sum()
    cum_hits = hits_sorted["_hits_clean"].cumsum()
    div_hits = tot_hits if tot_hits > 0 else 1.0
    hits_sorted["_cum_pct"] = cum_hits / div_hits
    hits_sorted["Hits Cumulative %"] = (hits_sorted["_cum_pct"] * 100.0).round(2)

    hits_sorted["Hits Class"] = [
        assign_tier(p, v, i) for i, (p, v) in enumerate(zip(hits_sorted["_cum_pct"], hits_sorted["_hits_clean"]))
    ]

    # Realign sorted series to original DataFrame index
    sales_indexed = sales_sorted.set_index("index")
    hits_indexed = hits_sorted.set_index("index")

    res_df["Sales Cumulative %"] = sales_indexed["Sales Cumulative %"]
    res_df["Sales Class"] = sales_indexed["Sales Class"]
    res_df["Hits Cumulative %"] = hits_indexed["Hits Cumulative %"]
    res_df["Hits Class"] = hits_indexed["Hits Class"]

    # Step 1c: Numerical Score Conversion: A = 4, B = 3, C = 2, D = 1
    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}
    res_df["Sales Score"] = res_df["Sales Class"].map(score_map)
    res_df["Hits Score"] = res_df["Hits Class"].map(score_map)

    # Step 2: Calculate Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)
    res_df["Combined Weighted Score"] = (
        sales_weight * res_df["Sales Score"] + hits_weight * res_df["Hits Score"]
    ).round(2)

    # Step 3: Assign Final Inventory Class based ONLY on Combined Weighted Score:
    # Class A: 3.25 - 4.00
    # Class B: 2.50 - 3.24
    # Class C: 1.75 - 2.49
    # Class D: < 1.75
    def map_final_class(score: float) -> str:
        if score >= 3.25:
            return "A"
        elif score >= 2.50:
            return "B"
        elif score >= 1.75:
            return "C"
        else:
            return "D"

    res_df["Final Inventory Class"] = res_df["Combined Weighted Score"].apply(map_final_class)
    res_df["Class"] = res_df["Final Inventory Class"]

    # Clean temporary calculation columns
    res_df = res_df.drop(columns=["_sales_clean", "_hits_clean"])

    # Step 4: Required Classification Summary
    # Report the number of items in each final category (A, B, C, and D) and total Sales
    total_skus = len(res_df)
    total_sales_val = float(tot_sales)
    total_hits_val = float(tot_hits)

    class_summary_rows = []
    class_counts: Dict[str, int] = {}
    category_sales_summary: Dict[str, Dict[str, Any]] = {}

    for c in ["A", "B", "C", "D"]:
        subset = res_df[res_df["Class"] == c]
        count = len(subset)
        class_counts[c] = count
        sku_pct = (count / total_skus * 100.0) if total_skus > 0 else 0.0

        sub_sales = float(
            pd.to_numeric(
                subset[sales_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        sales_pct = (sub_sales / total_sales_val * 100.0) if total_sales_val > 0 else 0.0

        sub_hits = float(
            pd.to_numeric(
                subset[hits_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0.0).sum()
        )
        hits_pct = (sub_hits / total_hits_val * 100.0) if total_hits_val > 0 else 0.0

        class_summary_rows.append({
            "Final Category": c,
            "Item Count": count,
            "% of Items": round(sku_pct, 1),
            "Total Sales ($)": round(sub_sales, 2),
            "% of Sales": round(sales_pct, 1),
            "Total POS Hits": int(round(sub_hits)),
            "% of POS Hits": round(hits_pct, 1),
        })

        category_sales_summary[c] = {
            "item_count": count,
            "total_sales": round(sub_sales, 2),
            "pct_of_items": round(sku_pct, 1),
            "pct_of_sales": round(sales_pct, 1),
        }

    class_summary_df = pd.DataFrame(class_summary_rows)

    # Bi-Criteria 2D Cross-Tabulation Grid
    crosstab_df = pd.crosstab(
        res_df["Sales Class"],
        res_df["Hits Class"],
        margins=True,
        margins_name="Total",
    ).reindex(
        index=["A", "B", "C", "D", "Total"],
        columns=["A", "B", "C", "D", "Total"],
        fill_value=0,
    )

    return {
        "classified_df": res_df,
        "class_summary": class_summary_df,
        "category_sales_summary": category_sales_summary,
        "crosstab": crosstab_df,
        "total_skus": total_skus,
        "total_sales": total_sales_val,
        "total_hits": total_hits_val,
        "class_counts": class_counts,
        "sales_col": sales_col,
        "hits_col": hits_col,
    }
