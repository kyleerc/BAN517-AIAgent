"""Supply Chain Analytics Inventory Decision Agent

Implements the 6-stage architecture:
Customer Question
  -> 1. Identify Model
  -> 2. Extract Inputs
  -> 3. Select Python Tool
  -> 4. Calculate
  -> 5. Verify
  -> 6. Explain

Supports:
1. Economic Order Quantity (EOQ)
2. Quantity Discount
3. Safety Stock and Reorder Point (Continuous Review)
4. Fixed-Period Review (Periodic Review / Order-Up-To Level)
5. Newsvendor (Single-Period Probabilistic)
"""

from __future__ import annotations

import enum
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import inventory_tools


class ModelType(str, enum.Enum):
    EOQ = "EOQ"
    QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    NEWSVENDOR = "NEWSVENDOR"
    UNKNOWN = "UNKNOWN"


MODEL_DISPLAY_NAMES = {
    ModelType.EOQ: "Economic Order Quantity (EOQ)",
    ModelType.QUANTITY_DISCOUNT: "Quantity Discount Model",
    ModelType.SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
    ModelType.FIXED_PERIOD_REVIEW: "Fixed-Period Review (Order-Up-To Policy)",
    ModelType.NEWSVENDOR: "Newsvendor Model (Single-Period Probabilistic)",
    ModelType.UNKNOWN: "Unknown / Unclassified Inventory Model",
}

TOOL_MAPPING: Dict[ModelType, Tuple[Callable[..., Dict[str, Any]], str]] = {
    ModelType.EOQ: (inventory_tools.calculate_eoq, "inventory_tools.calculate_eoq"),
    ModelType.QUANTITY_DISCOUNT: (inventory_tools.calculate_quantity_discount, "inventory_tools.calculate_quantity_discount"),
    ModelType.SAFETY_STOCK_ROP: (inventory_tools.calculate_safety_stock_rop, "inventory_tools.calculate_safety_stock_rop"),
    ModelType.FIXED_PERIOD_REVIEW: (inventory_tools.calculate_fixed_period_review, "inventory_tools.calculate_fixed_period_review"),
    ModelType.NEWSVENDOR: (inventory_tools.calculate_newsvendor, "inventory_tools.calculate_newsvendor"),
}


@dataclass
class StageTrace:
    stage_name: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    question: str
    model_type: ModelType
    model_name: str
    success: bool
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: Optional[str] = None
    calculation_results: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    step_by_step: List[str] = field(default_factory=list)
    explanation: str = ""
    error_message: Optional[str] = None
    stage_traces: List[StageTrace] = field(default_factory=list)
    raw_tool_output: Optional[Dict[str, Any]] = None


class InventoryDecisionAgent:
    """Agent orchestrating the 6-stage Supply Chain Decision Pipeline."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def run(self, question: str) -> AgentResponse:
        """Execute the end-to-end 6-stage decision pipeline."""
        traces: List[StageTrace] = []

        # -------------------------------------------------------------
        # Stage 1: Identify Model
        # -------------------------------------------------------------
        model_type, rationale = self._identify_model(question)
        model_name = MODEL_DISPLAY_NAMES.get(model_type, "Unknown")
        traces.append(
            StageTrace(
                stage_name="1. Identify Model",
                summary=f"Identified Model: {model_name}",
                details={"model_type": model_type.value, "rationale": rationale},
            )
        )

        if model_type == ModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                success=False,
                extracted_inputs={},
                missing_inputs=[],
                error_message=(
                    "Could not confidently determine the inventory model from the question. "
                    "Please specify whether your question concerns: "
                    "1) EOQ, 2) Quantity Discounts, 3) Safety Stock & Reorder Point, "
                    "4) Fixed-Period Review (Order-Up-To), or 5) Newsvendor."
                ),
                explanation=(
                    "### Model Classification Unresolved\n"
                    "The agent could not map your query to one of the 5 supported inventory models.\n\n"
                    "**Supported Models:**\n"
                    "- **EOQ**: For steady demand with ordering and holding costs.\n"
                    "- **Quantity Discount**: When purchase price drops at higher order volumes.\n"
                    "- **Safety Stock & ROP**: Continuous review buffering against demand/lead-time variability.\n"
                    "- **Fixed-Period Review**: Periodic order-up-to policy with fixed review intervals.\n"
                    "- **Newsvendor**: Single-period, perishable or seasonal items with salvage value and stockout costs."
                ),
                stage_traces=traces,
            )

        # -------------------------------------------------------------
        # Stage 2: Extract Inputs
        # -------------------------------------------------------------
        extracted_inputs, missing_inputs = self._extract_inputs(question, model_type)
        traces.append(
            StageTrace(
                stage_name="2. Extract Inputs",
                summary=f"Extracted {len(extracted_inputs)} inputs. Missing: {len(missing_inputs)}",
                details={"extracted": extracted_inputs, "missing": missing_inputs},
            )
        )

        if missing_inputs:
            guidance = self._generate_missing_input_guidance(model_type, extracted_inputs, missing_inputs)
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                success=False,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                error_message=f"Missing required parameter(s): {', '.join(missing_inputs)}",
                explanation=guidance,
                stage_traces=traces,
            )

        # -------------------------------------------------------------
        # Stage 3: Select Python Tool
        # -------------------------------------------------------------
        tool_func, tool_name = TOOL_MAPPING[model_type]
        traces.append(
            StageTrace(
                stage_name="3. Select Python Tool",
                summary=f"Selected Tool: {tool_name}",
                details={"tool_name": tool_name},
            )
        )

        # -------------------------------------------------------------
        # Stage 4: Calculate
        # -------------------------------------------------------------
        try:
            raw_output = tool_func(**extracted_inputs)
            traces.append(
                StageTrace(
                    stage_name="4. Calculate",
                    summary="Numerical computation executed successfully using Python.",
                    details={"results": raw_output.get("results", {})},
                )
            )
        except Exception as e:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                success=False,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                error_message=f"Calculation error in {tool_name}: {str(e)}",
                explanation=f"### Mathematical Execution Error\nAn error occurred during calculations:\n`{str(e)}`",
                stage_traces=traces,
            )

        # -------------------------------------------------------------
        # Stage 5: Verify
        # -------------------------------------------------------------
        verification_data = raw_output.get("verification", {"is_valid": True, "checks": []})
        all_passed = verification_data.get("is_valid", True)
        traces.append(
            StageTrace(
                stage_name="5. Verify",
                summary=f"Verification Checks: {'All Passed' if all_passed else 'Alerts Found'}",
                details=verification_data,
            )
        )

        # -------------------------------------------------------------
        # Stage 6: Explain
        # -------------------------------------------------------------
        explanation = self._generate_explanation(
            question=question,
            model_type=model_type,
            model_name=model_name,
            inputs=extracted_inputs,
            tool_output=raw_output,
            verification=verification_data,
        )
        traces.append(
            StageTrace(
                stage_name="6. Explain",
                summary="Synthesized executive explanation, mathematical breakdown, and managerial insights.",
                details={"recommendation": raw_output.get("recommendation", "")},
            )
        )

        return AgentResponse(
            question=question,
            model_type=model_type,
            model_name=model_name,
            success=True,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_results=raw_output.get("results"),
            verification=verification_data,
            step_by_step=raw_output.get("step_by_step", []),
            explanation=explanation,
            stage_traces=traces,
            raw_tool_output=raw_output,
        )

    # =========================================================================
    # Stage 1: Identification Logic
    # =========================================================================

    def _identify_model(self, question: str) -> Tuple[ModelType, str]:
        q = question.lower()

        # 1. Quantity Discount
        qd_keywords = [
            "discount", "price break", "volume discount", "quantity discount",
            "tier", "tiers", "bracket", "bulk discount", "all-units discount",
            "quantity break", "price schedule", "order quantity break"
        ]
        if any(k in q for k in qd_keywords) and ("tier" in q or "price" in q or "$" in q or "cost" in q or "break" in q or "unit" in q or "schedule" in q):
            return ModelType.QUANTITY_DISCOUNT, "Detected quantity discount / price-break terminology."

        # 2. Fixed-Period Review (Periodic Review)
        fpr_keywords = [
            "fixed period", "fixed-period", "periodic review", "order-up-to", "order up to",
            "target inventory", "target level", "review period", "every week", "every month",
            "every 2 weeks", "review interval", "on hand and on order", "inventory position"
        ]
        if any(k in q for k in fpr_keywords) and ("review" in q or "up to" in q or "target" in q):
            return ModelType.FIXED_PERIOD_REVIEW, "Detected periodic review / order-up-to inventory policy terms."

        # 3. Newsvendor
        nv_keywords = [
            "newsvendor", "newsboy", "single period", "single-period", "salvage", "overage",
            "underage", "critical ratio", "perishable", "seasonal fashion", "shortage penalty",
            "spoilage", "unsold units"
        ]
        if any(k in q for k in nv_keywords) or ("salvage" in q and ("price" in q or "cost" in q)):
            return ModelType.NEWSVENDOR, "Detected single-period probabilistic / newsvendor terms (salvage, underage, overage)."

        # 4. Safety Stock & Reorder Point
        rop_keywords = [
            "reorder point", "rop", "safety stock", "lead time demand", "service level",
            "cycle service level", "stockout risk", "continuous review", "(s, q)"
        ]
        if any(k in q for k in rop_keywords):
            return ModelType.SAFETY_STOCK_ROP, "Detected safety stock / reorder point continuous review terms."

        # 5. EOQ
        eoq_keywords = [
            "eoq", "economic order quantity", "wilson", "how much should we order",
            "optimal order quantity", "order setup cost", "holding cost per unit", "carrying cost"
        ]
        if any(k in q for k in eoq_keywords):
            return ModelType.EOQ, "Detected classic Economic Order Quantity (EOQ) setup & holding cost terms."

        # Secondary fallback heuristics
        if "order cost" in q and "holding" in q and "demand" in q:
            return ModelType.EOQ, "Query contains standard EOQ parameters (demand, order cost, holding cost)."
        if "lead time" in q and ("std" in q or "deviation" in q or "service level" in q):
            return ModelType.SAFETY_STOCK_ROP, "Query contains lead time and demand variability indicators."

        return ModelType.UNKNOWN, "No dominant inventory model keywords identified."

    # =========================================================================
    # Stage 2: Comprehensive Natural-Language Input Extraction Logic
    # =========================================================================

    def _extract_inputs(self, question: str, model_type: ModelType) -> Tuple[Dict[str, Any], List[str]]:
        """Examine the entire customer question and extract domain parameters with full synonym support."""
        # Remove commas inside numbers (e.g., "10,000" -> "10000") to preserve clause-separating punctuation
        q = re.sub(r"(?<=\d),(?=\d)", "", question)

        if model_type == ModelType.EOQ:
            return self._extract_eoq_inputs(q)
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(q)
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            return self._extract_rop_inputs(q)
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            return self._extract_fixed_period_review_inputs(q)
        elif model_type == ModelType.NEWSVENDOR:
            return self._extract_newsvendor_inputs(q)

        return {}, ["model parameters (Unknown model)"]

    def _extract_eoq_inputs(self, q: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Annual Demand (D)
        demand = self._extract_first_number(
            q,
            [
                r"(?:annual\s*demand|annual\s*usage|annual\s*sales|annual\s*consumption)\s*(?:rate)?\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:rate)?\s*(?:of|is|:|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|cases|boxes)?(?:\s*(?:per|/|a)\s*year|annually)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|cases|boxes)?\s*(?:per\s*year|/year|annually|a\s*year|annual\s*demand)",
                r"sells?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|cases|boxes)?(?:\s*(?:per|/|a)\s*year|annually)?",
                r"usage\s*(?:of|is)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?(?:\s*(?:per|/|a)\s*year|annually)?",
                r"demand\s*(?:of|is)\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        if demand is not None:
            extracted["demand"] = float(demand)
        else:
            missing.append("demand (Annual Demand D in units/year)")

        # 2. Ordering Cost (S) - Supports all synonyms:
        # ordering cost, order cost, setup cost, setup expense, cost per order,
        # cost of placing an order, ordering/setup cost, order fee, etc.
        order_cost = self._extract_first_number(
            q,
            [
                r"(?:ordering/setup|ordering\s*and\s*setup|setup\s*and\s*ordering|ordering|order|setup)\s*(?:cost|expense|fee)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"setup\s*expense\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:cost\s*per\s*order|cost\s*of\s*placing\s*(?:an|each|every)?\s*order|cost\s*to\s*place\s*(?:an|each|every)?\s*order|order\s*placement\s*cost)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s*place\s*(?:an|each|every)?\s*order|per\s*order|/order)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*order|/order|to\s*place\s*(?:an|each|every)?\s*order)",
                r"incurs\s*(?:an\s*administrative\s*cost|a\s*fee|a\s*cost)\s*of\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*order)?",
            ],
        )
        if order_cost is not None:
            extracted["order_cost"] = float(order_cost)
        else:
            missing.append("order_cost (Ordering / Setup Cost S in $/order)")

        # 3. Holding Cost (H) - Fixed or Percentage
        holding_cost = self._extract_first_number(
            q,
            [
                r"(?:annual\s*)?(?:holding|carrying|inventory\s*carrying)\s*(?:cost|expense|fee)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
                r"(?:cost\s*to\s*hold|cost\s*of\s*holding|cost\s*to\s*carry|cost\s*of\s*carrying)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit\s*(?:per|/)\s*year|/unit/year|per\s*unit/yr|/unit/yr|per\s*item\s*per\s*year)",
                r"holding\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)(?!\s*%)",
            ],
        )
        if holding_cost is not None:
            extracted["holding_cost"] = float(holding_cost)
        else:
            # Check for percentage holding cost: e.g. 20% holding rate * unit cost
            h_pct = self._extract_first_number(
                q,
                [
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|annual\s*carrying|inventory\s*carrying)",
                    r"(?:holding|carrying)\s*(?:rate|percentage|cost\s*rate)\s*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ],
            )
            unit_c = self._extract_first_number(
                q,
                [
                    r"(?:unit\s*purchase\s*price|purchase\s*price|unit\s*cost|unit\s*price|item\s*cost|acquisition\s*cost)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                    r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each|to\s*buy|to\s*purchase)(?!\s*(?:per\s*year|/year))",
                    r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)(?!\s*(?:per\s*year|/year))",
                ],
            )
            if h_pct is not None and unit_c is not None:
                extracted["holding_cost"] = (float(h_pct) / 100.0) * float(unit_c)
                extracted["unit_cost"] = float(unit_c)
            else:
                missing.append("holding_cost (Annual Holding / Carrying Cost H in $/unit/year)")

        # Optional: Unit purchase acquisition cost
        if "unit_cost" not in extracted:
            unit_c = self._extract_first_number(
                q,
                [
                    r"(?:unit\s*purchase\s*price|purchase\s*price|unit\s*cost|unit\s*price|item\s*cost|acquisition\s*cost)\s*(?:of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                    r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)(?!\s*(?:per\s*year|/year))",
                ],
            )
            if unit_c is not None:
                extracted["unit_cost"] = float(unit_c)

        return extracted, missing

    def _extract_quantity_discount_inputs(self, q: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted: Dict[str, Any] = {}
        missing: List[str] = []

        # Demand
        demand = self._extract_first_number(
            q,
            [
                r"(?:annual\s*demand|annual\s*usage|annual\s*sales)\s*(?:rate)?\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"demand\s*(?:rate)?\s*(?:of|is|are|:|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|cases|boxes)?(?:\s*(?:per|/|a)\s*year|annually)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|cases|boxes)?\s*(?:per\s*year|/year|annually|a\s*year)",
            ],
        )
        if demand is not None:
            extracted["demand"] = float(demand)
        else:
            missing.append("demand (Annual Demand D in units)")

        # Ordering Cost (supports all EOQ ordering cost synonyms)
        order_cost = self._extract_first_number(
            q,
            [
                r"(?:ordering/setup|ordering\s*and\s*setup|setup\s*and\s*ordering|ordering|order|setup)\s*(?:cost|expense|fee)\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"setup\s*expense\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"(?:cost\s*per\s*order|cost\s*of\s*placing\s*(?:an|each|every)?\s*order|cost\s*to\s*place\s*(?:an|each|every)?\s*order)\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s*place\s*(?:an|each|every)?\s*order|per\s*order|/order)",
                r"incurs\s*(?:an\s*administrative\s*cost|a\s*fee|a\s*cost)\s*of\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*order|/order)",
            ],
        )
        if order_cost is not None:
            extracted["order_cost"] = float(order_cost)
        else:
            missing.append("order_cost (Cost per order S in $)")

        # Holding Cost: Percentage holding cost vs Fixed holding cost
        h_rate = self._extract_first_number(
            q,
            [
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|annual\s*carrying)",
                r"(?:annual\s*)?(?:holding|carrying)\s*(?:cost\s*)?(?:rate|percentage)?\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                r"(?:percentage\s*holding\s*cost|holding\s*percentage|holding\s*cost\s*percentage)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
                r"carrying\s*rate\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        fixed_h = self._extract_first_number(
            q,
            [
                r"fixed\s*holding\s*cost\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"holding\s*cost\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|/unit)(?!\s*%)",
            ],
        )
        if h_rate is not None:
            val = float(h_rate)
            extracted["holding_cost_rate"] = val / 100.0 if val >= 1.0 else val
        elif fixed_h is not None:
            extracted["fixed_holding_cost"] = float(fixed_h)
        else:
            missing.append("holding_cost (Either percentage holding_cost_rate or fixed_holding_cost)")

        # Base / unit purchase price if specified
        base_unit_price = self._extract_first_number(
            q,
            [
                r"(?:unit\s*purchase\s*price|purchase\s*price|base\s*unit\s*price)\s*(?:of|is|are|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        if base_unit_price is not None:
            extracted["base_unit_price"] = float(base_unit_price)

        # Discount Tiers / Quantity Breaks
        tiers = self._extract_tiers(q)
        if tiers:
            extracted["tiers"] = tiers
        else:
            missing.append("tiers (Quantity break tiers, e.g. '1-999 @ $5.00, 1000+ @ $4.80')")

        return extracted, missing

    def _extract_rop_inputs(self, q: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted: Dict[str, Any] = {}
        missing: List[str] = []
        q_work = q

        # Plain language assignment: treats "is", "is equal to", "equals", "equal to", ":", "=", "of", "averages" as equivalent assignment operators.
        assign_op = r"(?:is\s*(?:equal\s*to|set\s*to|set\s*at)?|equals|equal\s*to|has\s*an?\s*(?:average|mean)\s*(?:of)?|averages|are|of|:|=)"

        # 1. Lead Time & Lead-Time Standard Deviation
        # Combined pattern: e.g., "Average lead time is 3 weeks with a standard deviation of 0.5 weeks"
        # "lead time = 3 weeks, std dev is 0.5 weeks", "average lead time of 3 weeks and std dev of 0.5"
        lt_comb_m = re.search(
            r"(?:average|mean|expected)?\s*(?:supplier\s*)?lead\s*time\s*" + assign_op + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)?\s*(?:,|;)?\s*(?:\(?\s*(?:with|and)?\s*(?:a\s*)?(?:standard\s*deviation|std\s*dev)\s*" + assign_op + r"?)\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if lt_comb_m:
            extracted["lead_time"] = float(lt_comb_m.group(1))
            extracted["std_lead_time"] = float(lt_comb_m.group(2))
            q_work = q_work[:lt_comb_m.start()] + " [MASKED_LT_AND_STD] " + q_work[lt_comb_m.end():]

        # Standalone lead-time standard deviation if not captured in combined pattern
        if "std_lead_time" not in extracted:
            std_lt_m = re.search(
                r"(?:standard\s*deviation\s*of\s*(?:supplier\s*)?lead\s*time|(?:supplier\s*)?lead[- ]?time\s*(?:standard\s*deviation|std\s*dev)|sigma_l)\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
                q_work,
                re.IGNORECASE,
            )
            if std_lt_m:
                extracted["std_lead_time"] = float(std_lt_m.group(1))
                q_work = q_work[:std_lt_m.start()] + " [MASKED_STD_LT] " + q_work[std_lt_m.end():]
            else:
                extracted["std_lead_time"] = 0.0

        # 2. Demand & Demand Standard Deviation
        # Combined pattern: e.g., "average demand of 40 units per day with a standard deviation of 8 units per day"
        # "average demand is 40 units per day with a standard deviation of 8 units"
        d_comb_m = re.search(
            r"(?:average|mean|expected|daily|weekly|periodic)?\s*demand\s*(?:rate)?\s*" + assign_op + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units?|boxes?|cases?|items?)?(?:\s*(?:per|/|a)\s*(?:day|week|month))?\s*(?:,|;)?\s*(?:\(?\s*(?:with|and)?\s*(?:a\s*)?(?:standard\s*deviation|std\s*dev)\s*" + assign_op + r"?)\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if d_comb_m:
            extracted["avg_demand"] = float(d_comb_m.group(1))
            extracted["std_demand"] = float(d_comb_m.group(2))
            q_work = q_work[:d_comb_m.start()] + " [MASKED_D_AND_STD] " + q_work[d_comb_m.end():]

        # Standalone demand standard deviation if not captured in combined pattern
        if "std_demand" not in extracted:
            std_d_m = re.search(
                r"(?:standard\s*deviation\s*of\s*(?:daily|weekly|periodic)?\s*demand|demand\s*(?:standard\s*deviation|std\s*dev)|sigma_d)\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
                q_work,
                re.IGNORECASE,
            )
            if std_d_m:
                extracted["std_demand"] = float(std_d_m.group(1))
                q_work = q_work[:std_d_m.start()] + " [MASKED_STD_D] " + q_work[std_d_m.end():]
            else:
                m_sec = re.search(
                    r"(?<!lead[- ]time\s)standard\s*deviation\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units?|items?|cases?|boxes?)",
                    q_work,
                    re.IGNORECASE,
                )
                if m_sec:
                    extracted["std_demand"] = float(m_sec.group(1))
                    q_work = q_work[:m_sec.start()] + " [MASKED_STD_D] " + q_work[m_sec.end():]
                else:
                    extracted["std_demand"] = 0.0

        # 3. Average Lead Time (L) - extracted from q_work
        # Explicitly recognizes natural language including "average lead time is" (where "is" = "=")
        if "lead_time" not in extracted:
            lt_patterns = [
                r"(?:average|mean|expected)?\s*(?:supplier\s*)?lead\s*time\s*" + assign_op + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)?",
                r"(?:average|mean|expected)?\s*(?:supplier\s*)?lead\s*time\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:day|week|month|period)s?\s*(?:of\s*)?(?:supplier\s*)?(?:lead\s*time|delivery\s*time|replenishment\s*time)",
            ]
            for pat in lt_patterns:
                lt_m = re.search(pat, q_work, re.IGNORECASE)
                if lt_m and lt_m.group(1):
                    extracted["lead_time"] = float(lt_m.group(1))
                    q_work = q_work[:lt_m.start()] + " [MASKED_LT] " + q_work[lt_m.end():]
                    break
            if "lead_time" not in extracted:
                missing.append("lead_time (Average lead time L in days/weeks)")

        # 4. Average Demand (d) - extracted from q_work
        # Explicitly recognizes natural language including "demand is", "average demand is" (where "is" = "=")
        if "avg_demand" not in extracted:
            d_patterns = [
                r"(?:average|mean|expected|daily|weekly|periodic)?\s*demand\s*(?:rate)?\s*" + assign_op + r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units?|boxes?|cases?|items?)?(?:\s*(?:per|/|a)\s*(?:day|week|month))?",
                r"(?:average|mean|expected)\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units?|boxes?|cases?|items?)?\s*(?:per|/|a)\s*(?:day|week|month)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:units?|boxes?|cases?|items?)\s*(?:per|/|a)\s*(?:day|week|month)",
            ]
            for pat in d_patterns:
                d_m = re.search(pat, q_work, re.IGNORECASE)
                if d_m and d_m.group(1):
                    extracted["avg_demand"] = float(d_m.group(1))
                    q_work = q_work[:d_m.start()] + " [MASKED_D] " + q_work[d_m.end():]
                    break
            if "avg_demand" not in extracted:
                missing.append("avg_demand (Average demand per period d)")

        # 5. Explicit Safety Stock if directly specified
        # Recognizes "safety stock is", "safety stock =", "safety stock of", etc.
        explicit_ss = self._extract_first_number(
            q_work,
            [
                r"(?:safety\s*stock|safety\s*buffer)\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
                r"maintain\s*(?:a\s*)?safety\s*stock\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        if explicit_ss is not None:
            extracted["safety_stock"] = float(explicit_ss)

        # 6. Service Level or Z-Score
        # Recognizes "service level is", "service level =", "service level of", etc.
        service_level = self._extract_first_number(
            q_work,
            [
                r"(?:cycle\s*)?service\s*level\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?service\s*level",
                r"csl\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        z_score = self._extract_first_number(
            q_work,
            [
                r"z[- ]?score\s*" + assign_op + r"?\s*([0-9]+(?:\.[0-9]+)?)",
                r"z\s*" + assign_op + r"\s*([0-9]+(?:\.[0-9]+)?)",
            ],
        )
        if service_level is not None:
            val = float(service_level)
            extracted["service_level"] = val / 100.0 if val > 1.0 else val
        elif z_score is not None:
            extracted["z_score"] = float(z_score)
        elif explicit_ss is not None:
            pass  # Explicit safety stock provided, service level not required
        elif extracted.get("std_demand", 0) > 0 or extracted.get("std_lead_time", 0) > 0:
            missing.append("service_level (Target cycle service level e.g. 95% or z-score)")
        else:
            extracted["service_level"] = 1.0

        return extracted, missing

    def _extract_fixed_period_review_inputs(self, q: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted: Dict[str, Any] = {}
        missing: List[str] = []
        q_work = q

        # 1. Review Period (R)
        rp_m = re.search(
            r"(?:review\s*period|review\s*interval)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)?",
            q_work,
            re.IGNORECASE,
        )
        if not rp_m:
            rp_m = re.search(
                r"review(?:ed)?\s*every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)",
                q_work,
                re.IGNORECASE,
            )
        if rp_m:
            extracted["review_period"] = float(rp_m.group(1))
            q_work = q_work[:rp_m.start()] + " [MASKED_RP] " + q_work[rp_m.end():]
        else:
            missing.append("review_period (Review period interval R in days/weeks)")

        # 2. Lead Time (L)
        lt_m = re.search(
            r"(?:delivery\s*time|replenishment\s*time|lead\s*time)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|periods?)?",
            q_work,
            re.IGNORECASE,
        )
        if not lt_m:
            lt_m = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)\s*(?:lead\s*time|delivery\s*time)",
                q_work,
                re.IGNORECASE,
            )
        if lt_m:
            extracted["lead_time"] = float(lt_m.group(1))
            q_work = q_work[:lt_m.start()] + " [MASKED_LT] " + q_work[lt_m.end():]
        else:
            missing.append("lead_time (Delivery lead time L in days/weeks)")

        # 3. Demand Standard Deviation (sigma_d)
        sd_m = re.search(
            r"(?:standard\s*deviation\s*(?:of\s*(?:daily|weekly|periodic)?\s*demand)?|demand\s*std(?: dev)?)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if sd_m:
            extracted["std_demand"] = float(sd_m.group(1))
            q_work = q_work[:sd_m.start()] + " [MASKED_SD] " + q_work[sd_m.end():]
        else:
            extracted["std_demand"] = 0.0

        # 4. Demand (d)
        d_m = re.search(
            r"(?:average|mean|expected|daily|weekly|periodic)?\s*demand\s*(?:rate)?\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|cases|boxes|items)?(?:\s*(?:per|/|a)\s*(?:day|week|month))?",
            q_work,
            re.IGNORECASE,
        )
        if d_m and d_m.group(1):
            extracted["avg_demand"] = float(d_m.group(1))
            q_work = q_work[:d_m.start()] + " [MASKED_D] " + q_work[d_m.end():]
        else:
            missing.append("avg_demand (Average periodic demand rate d)")

        # 5. Explicit Safety Stock if given
        ss_m = re.search(
            r"(?:safety\s*stock|safety\s*buffer)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if ss_m:
            extracted["safety_stock"] = float(ss_m.group(1))
            q_work = q_work[:ss_m.start()] + " [MASKED_SS] " + q_work[ss_m.end():]

        # 6. Service Level
        sl_m = re.search(
            r"(?:cycle\s*)?service\s*level\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            q_work,
            re.IGNORECASE,
        )
        if sl_m:
            val = float(sl_m.group(1))
            extracted["service_level"] = val / 100.0 if val > 1.0 else val
        elif "safety_stock" in extracted:
            extracted["service_level"] = 1.0
        else:
            extracted["service_level"] = 0.95

        # 7. Inventory Position (Direct or through On-Hand, On-Order, Backorders)
        ip_m = re.search(
            r"(?:inventory\s*position|effective\s*inventory\s*position)\s*(?:of|is|are|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if ip_m:
            extracted["inventory_position"] = float(ip_m.group(1))
            extracted["current_inventory"] = float(ip_m.group(1))
        else:
            on_hand = self._extract_first_number(
                q_work,
                [
                    r"(?:current\s*inventory\s*on[- ]?hand|inventory\s*on[- ]?hand|on[- ]?hand\s*inventory|stock\s*on\s*hand)\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                    r"(?:currently\s+we\s+have|we\s+have|currently|current)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases|boxes)?\s*on[- ]?hand",
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases|boxes)?\s*(?:on[- ]?hand|in\s*stock)",
                    r"on[- ]?hand\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                ],
            )
            on_order = self._extract_first_number(
                q_work,
                [
                    r"(?:on[- ]?order|in[- ]?transit|pipeline\s*stock)\s*(?:inventory)?\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|cases|boxes)?\s*(?:in[- ]?transit|on[- ]?order)",
                ],
            )
            backorders = self._extract_first_number(
                q_work,
                [
                    r"(?:backorders?|backlog)\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:backorders?|backlog)",
                ],
            )
            if backorders is None and any(w in q.lower() for w in ["zero backorder", "no backorder", "0 backorder"]):
                backorders = 0.0

            if on_hand is not None:
                extracted["current_inventory"] = float(on_hand)
                extracted["on_order"] = float(on_order) if on_order is not None else 0.0
                extracted["backorders"] = float(backorders) if backorders is not None else 0.0
            else:
                missing.append("current_inventory (Current stock on hand or inventory position)")

        return extracted, missing

    def _extract_newsvendor_inputs(self, q: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted: Dict[str, Any] = {}
        missing: List[str] = []
        q_work = q

        # 1. Underage / Shortage Cost (Cu)
        cu_m = re.search(
            r"(?:underage\s*cost|cost\s*of\s*underage|unit\s*underage\s*cost|shortage\s*cost|cost\s*of\s*shortage)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        cu = float(cu_m.group(1)) if cu_m else None
        if cu_m:
            q_work = q_work[:cu_m.start()] + " [MASKED_CU] " + q_work[cu_m.end():]

        # 2. Overage / Excess Cost (Co)
        co_m = re.search(
            r"(?:overage\s*cost|cost\s*of\s*overage|unit\s*overage\s*cost|excess\s*cost|cost\s*of\s*excess)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        co = float(co_m.group(1)) if co_m else None
        if co_m:
            q_work = q_work[:co_m.start()] + " [MASKED_CO] " + q_work[co_m.end():]

        # 3. Selling price (p)
        sp_m = re.search(
            r"(?:sell(?:ing)?\s*price|retail\s*price|sells?\s*for|price\s*per\s*unit)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        selling_price = float(sp_m.group(1)) if sp_m else None
        if sp_m:
            q_work = q_work[:sp_m.start()] + " [MASKED_SP] " + q_work[sp_m.end():]

        # 4. Purchase Cost (c)
        cost_m = re.search(
            r"(?<!holding\s)(?<!carrying\s)(?<!order\s)(?<!setup\s)(?:procurement\s*cost|purchase\s*cost|wholesale\s*cost|wholesale\s*price|unit\s*cost|purchase\s*price|acquisition\s*cost|cost)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if not cost_m:
            cost_m = re.search(
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s*buy|to\s*produce|to\s*make|to\s*manufacture|to\s*procure)",
                q_work,
                re.IGNORECASE,
            )
        cost = float(cost_m.group(1)) if cost_m else None
        if cost_m:
            q_work = q_work[:cost_m.start()] + " [MASKED_COST] " + q_work[cost_m.end():]

        # 5. Salvage Value (s)
        salvage_m = re.search(
            r"(?:salvage\s*(?:value|price)?|salvaged\s*for|scrap\s*(?:value|price)?|clearance\s*price|liquidated\s*at)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        salvage = float(salvage_m.group(1)) if salvage_m else None
        if salvage is None and any(w in q.lower() for w in ["no salvage", "zero salvage", "worthless", "discarded", "thrown away", "no salvage value"]):
            salvage = 0.0

        # Shortage penalty / goodwill loss (g)
        pen_m = re.search(
            r"(?:shortage|goodwill|lost\s*sales?)\s*penalty\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        penalty = float(pen_m.group(1)) if pen_m else 0.0

        # 6. Demand Standard Deviation (sigma)
        std_d_m = re.search(
            r"(?:standard\s*deviation\s*(?:of\s*demand)?|demand\s*(?:standard\s*deviation|std\s*dev)|sigma)\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        std_d = float(std_d_m.group(1)) if std_d_m else None
        if std_d_m:
            q_work = q_work[:std_d_m.start()] + " [MASKED_STD_D] " + q_work[std_d_m.end():]

        # 7. Mean Demand (mu)
        mean_d_m = re.search(
            r"(?:mean|expected|average)\s*demand\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            q_work,
            re.IGNORECASE,
        )
        if not mean_d_m:
            mean_d_m = re.search(
                r"demand\s*(?:is\s*normally\s*distributed\s*with\s*)?mean\s*(?:is|are|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
                q_work,
                re.IGNORECASE,
            )
        mean_d = float(mean_d_m.group(1)) if mean_d_m else None

        # Populate
        if mean_d is not None:
            extracted["mean_demand"] = mean_d
        else:
            missing.append("mean_demand (Expected demand mu)")

        if std_d is not None:
            extracted["std_demand"] = std_d
        else:
            missing.append("std_demand (Demand standard deviation sigma)")

        if cu is not None and co is not None:
            extracted["underage_cost"] = cu
            extracted["overage_cost"] = co
            extracted["selling_price"] = cu + co
            extracted["cost"] = co
            extracted["salvage_value"] = 0.0
        elif selling_price is not None and cost is not None and salvage is not None:
            extracted["selling_price"] = selling_price
            extracted["cost"] = cost
            extracted["salvage_value"] = salvage
            extracted["penalty"] = penalty
        else:
            if selling_price is None and cu is None:
                missing.append("selling_price (Unit selling price p)")
            if cost is None and co is None:
                missing.append("cost (Unit procurement cost c)")
            if salvage is None and co is None:
                missing.append("salvage_value (Salvage recovery value s)")

        return extracted, missing

    def _extract_first_number(self, text: str, patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                try:
                    raw_str = match.group(1).rstrip(".")
                    return float(raw_str)
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_tiers(self, text: str) -> List[Dict[str, Any]]:
        """Extract multi-tier discount schedules from text."""
        tiers: List[Dict[str, Any]] = []

        # Pattern 1: e.g. "1-999 units @ $5.00" or "1 to 999 units the unit price is $5.00"
        p1 = re.findall(
            r"(?:tier\s*\d+[:\s]*)?(?:orders?\s*(?:of)?\s*)?([0-9]+)\s*(?:-|to)\s*([0-9]+)\s*(?:units?)?\s*(?:@|at|:|=|the\s*unit\s*price\s*is|unit\s*price\s*(?:is|are|:|=)?|price\s*(?:is|are|:|=)?)\s*\$?([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        for m in p1:
            tiers.append({"min_qty": float(m[0]), "max_qty": float(m[1]), "unit_price": float(m[2])})

        # Pattern 2: e.g. "1000+ units @ $4.80" or "1000 or more units unit price is $4.80"
        p2 = re.findall(
            r"(?:tier\s*\d+[:\s]*)?(?:orders?\s*(?:of)?\s*)?([0-9]+)\s*(?:\+|or\s*more|and\s*above|and\s*over)\s*(?:units?)?\s*(?:@|at|:|=|the\s*unit\s*price\s*is|unit\s*price\s*(?:is|are|:|=)?|price\s*(?:is|are|:|=)?)\s*\$?([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE,
        )
        for m in p2:
            tiers.append({"min_qty": float(m[0]), "max_qty": None, "unit_price": float(m[1])})

        # Price-first pattern: "$5.00 for 1-999 units, $4.80 for 1000+ units"
        if not tiers:
            p3 = re.findall(
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?\s*for\s*(?:orders?\s*(?:of)?\s*)?([0-9]+)\s*(?:-|to)\s*([0-9]+)\s*(?:units?)?",
                text,
                re.IGNORECASE,
            )
            for m in p3:
                tiers.append({"min_qty": float(m[1]), "max_qty": float(m[2]), "unit_price": float(m[0])})

            p4 = re.findall(
                r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?\s*for\s*(?:orders?\s*(?:of)?\s*)?([0-9]+)\s*(?:\+|or\s*more|and\s*above|and\s*over)\s*(?:units?)?",
                text,
                re.IGNORECASE,
            )
            for m in p4:
                tiers.append({"min_qty": float(m[1]), "max_qty": None, "unit_price": float(m[0])})

        # Fallback generic range match: "< 500 @ $50, >= 500 @ $45"
        if not tiers:
            p5 = re.findall(
                r"(?:tier\s*\d+:?\s*)?(?:<\s*([0-9]+)|([0-9]+)\s*-\s*([0-9]+)|>=?\s*([0-9]+))\s*(?:units)?\s*(?:@|at|for|:)\s*\$?([0-9]+(?:\.[0-9]+)?)",
                text,
                re.IGNORECASE,
            )
            for m in p5:
                price = float(m[4])
                if m[0]:
                    tiers.append({"min_qty": 1.0, "max_qty": float(m[0]) - 1, "unit_price": price})
                elif m[1] and m[2]:
                    tiers.append({"min_qty": float(m[1]), "max_qty": float(m[2]), "unit_price": price})
                elif m[3]:
                    tiers.append({"min_qty": float(m[3]), "max_qty": None, "unit_price": price})

        tiers.sort(key=lambda x: x["min_qty"])
        return tiers

    # =========================================================================
    # Stage 6: Explanation & Reporting Logic
    # =========================================================================

    def _generate_missing_input_guidance(
        self, model_type: ModelType, extracted: Dict[str, Any], missing: List[str]
    ) -> str:
        model_name = MODEL_DISPLAY_NAMES.get(model_type, "Inventory Model")
        extracted_str = (
            "\n".join([f"- **{k}**: {v}" for k, v in extracted.items()])
            if extracted
            else "*(None extracted yet)*"
        )
        missing_str = "\n".join([f"- ⚠️ **{m}**" for m in missing])

        return (
            f"### Model Identified: {model_name}\n\n"
            f"To complete the calculation, additional parameters are required.\n\n"
            f"**Successfully Extracted Inputs:**\n{extracted_str}\n\n"
            f"**Missing Required Parameters:**\n{missing_str}\n\n"
            f"**Suggested Next Step:** Please reply providing the missing values."
        )

    def _generate_explanation(
        self,
        question: str,
        model_type: ModelType,
        model_name: str,
        inputs: Dict[str, Any],
        tool_output: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> str:
        recommendation = tool_output.get("recommendation", "")
        inputs_extracted = tool_output.get("inputs_extracted", tool_output.get("inputs", inputs))
        var_defs = tool_output.get("variable_definitions", {})
        formulas = tool_output.get("formula", {})
        formula_subs = tool_output.get("formula_with_substituted_values", {})
        intermediates = tool_output.get("intermediate_calculations", {})
        final_ans = tool_output.get("final_answer", {})
        checks = tool_output.get("verification_checks", verification.get("checks", []))
        assumptions = tool_output.get("assumptions", [])

        # Format 1. Inputs Extracted Table
        input_lines = []
        for k, v in inputs_extracted.items():
            if k == "tiers" and isinstance(v, list):
                tier_str = ", ".join([f"[{t.get('min_qty')}-{t.get('max_qty') or 'inf'}: ${t.get('unit_price')}]" for t in v])
                input_lines.append(f"| `{k}` | {tier_str} |")
            else:
                input_lines.append(f"| `{k}` | `{v}` |")
        inputs_table = "\n".join(input_lines) if input_lines else "| None | None |"

        # Format 2. Variable Definitions
        var_defs_lines = [f"- **`{k}`**: {v}" for k, v in var_defs.items()]
        var_defs_md = "\n".join(var_defs_lines)

        # Format 3. Formula (Symbolic)
        formula_lines = (
            [f"- **{k}**: `{v}`" for k, v in formulas.items()]
            if isinstance(formulas, dict)
            else [f"- `{f}`" for f in formulas]
        )
        formula_md = "\n".join(formula_lines)

        # Format 4. Formula with Substituted Values
        sub_lines = (
            [f"- **{k}**: {v}" for k, v in formula_subs.items()]
            if isinstance(formula_subs, dict)
            else [f"- {s}" for s in formula_subs]
        )
        formula_subs_md = "\n".join(sub_lines)

        # Format 5. Intermediate Calculations
        intermediate_lines = []
        for k, v in intermediates.items():
            if k in ("evaluated_tiers", "best_tier"):
                continue
            if isinstance(v, float):
                intermediate_lines.append(f"| `{k}` | `{v:,.4f}` |")
            else:
                intermediate_lines.append(f"| `{k}` | `{v}` |")
        intermediates_table = "\n".join(intermediate_lines) if intermediate_lines else "| None | None |"

        # Format 6. Final Answer
        final_ans_lines = []
        if isinstance(final_ans, dict):
            for k, v in final_ans.items():
                if isinstance(v, float):
                    final_ans_lines.append(f"- **{k.replace('_', ' ').title()}**: `{v:,.2f}`")
                else:
                    final_ans_lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
        else:
            final_ans_lines.append(f"- **Result**: `{final_ans}`")
        final_ans_md = "\n".join(final_ans_lines)

        # Format 7. Verification Checks
        all_passed = all(c.get("passed", False) for c in checks) if checks else True
        status_badge = "✅ **ALL CHECKS PASSED**" if all_passed else "⚠️ **CHECKS FLAGGED ANOMALY**"
        checks_lines = [
            f"- {'✅' if c.get('passed') else '❌'} **{c.get('check')}**: {c.get('details')}"
            for c in checks
        ]
        checks_md = "\n".join(checks_lines)

        # Format 8. Assumptions
        assumptions_lines = [f"- {a}" for a in assumptions]
        assumptions_md = "\n".join(assumptions_lines)

        md = f"""# Supply Chain Student Verification Report: {model_name}

> **Core Decision**: {recommendation}

---

### 1. Inputs Extracted
| Parameter | Extracted Numerical Value |
| :--- | :--- |
{inputs_table}

---

### 2. Variable Definitions
{var_defs_md}

---

### 3. Formula (Symbolic)
{formula_md}

---

### 4. Formula with Substituted Values
{formula_subs_md}

---

### 5. Intermediate Calculations
| Intermediate Metric | Computed Numerical Value |
| :--- | :--- |
{intermediates_table}

---

### 6. Final Answer
{final_ans_md}

---

### 7. Verification Checks ({status_badge})
{checks_md}

---

### 8. Assumptions
{assumptions_md}
"""
        return md

    def _get_managerial_insights(self, model_type: ModelType, results: Dict[str, Any]) -> str:
        if model_type == ModelType.EOQ:
            return (
                "- **Cost Tradeoff**: EOQ operates at the exact equilibrium where annual order setup costs equal holding carrying costs. "
                "- **Robustness & Sensitivity**: The total cost curve is flat near the minimum; a 20% variance in order batch size increases total relevant inventory cost by less than 2%.\n"
                "- **Cash Flow Impact**: Consolidating purchase batches reduces transaction overhead but ties up working capital in cycle stock."
            )
        elif model_type == ModelType.QUANTITY_DISCOUNT:
            opt_tier = results.get("optimal_tier", 1)
            return (
                f"- **Discount Realization**: Tier {opt_tier} was selected because unit purchasing savings surpass the elevated holding cost of carrying larger batch sizes.\n"
                "- **Working Capital Caution**: Qualifying for bulk price breaks requires holding larger average cycle inventory; ensure warehouse capacity and shelf-life are adequate.\n"
                "- **Supplier Alignment**: Consider negotiating smaller delivery lot releases against a blanket purchase order to capture bulk pricing without warehouse strain."
            )
        elif model_type == ModelType.SAFETY_STOCK_ROP:
            csl = results.get("service_level", 0.95) * 100
            ss = results.get("safety_stock", 0)
            return (
                f"- **Service Level Non-Linearity**: Maintaining a {csl:.1f}% service level requires {ss:.1f} units of safety buffer. Note that increasing from 95% to 99% increases required safety stock by ~41% due to the normal distribution tail.\n"
                "- **Lead Time Compression**: Reducing supplier lead time has a square-root impact on required safety stock, freeing up cash.\n"
                "- **Continuous Monitoring**: An automated alert should trigger when physical stock reaches the Reorder Point."
            )
        elif model_type == ModelType.FIXED_PERIOD_REVIEW:
            prot = results.get("protection_interval", 0)
            return (
                f"- **Periodic Review Overhead**: Because inventory is only reviewed periodically, the buffer must cover the entire protection window of (Review Period + Lead Time = {prot:.1f} periods).\n"
                "- **Transportation Efficiency**: Periodic review shines when coordinating deliveries across multiple SKUs on scheduled trucking runs, lowering logistics costs."
            )
        elif model_type == ModelType.NEWSVENDOR:
            cr = results.get("critical_ratio", 0.5) * 100
            return (
                f"- **Critical Fractile Dynamics**: The critical ratio is {cr:.1f}%. When the underage margin exceeds overage risk (CR > 50%), the optimal strategy intentionally over-stocks relative to the mean demand.\n"
                "- **Secondary Market Value**: Negotiating higher salvage returns directly raises the critical ratio and unlocks higher order volumes and sales fulfillment."
            )
        return "- Align inventory decisions with overall corporate working capital and service level targets."
