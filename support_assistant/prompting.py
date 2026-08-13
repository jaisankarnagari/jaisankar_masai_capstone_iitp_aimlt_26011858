"""
Structured Prompt Templates for Zepto Support Assistant

This module provides reusable prompt templates following the role-context-task-format-length
skeleton with explicit negative constraints and few-shot examples.
"""


class ZeptoPromptTemplate:
    """Structured prompt builder for Zepto support queries"""

    SYSTEM_PROMPT = """# ROLE
You are a Zepto Customer Support Assistant with expertise in:
- Zepto company policies and procedures
- Order processing and delivery management
- Refund and return procedures
- Customer service best practices
- Customer data privacy and security

# CONTEXT
Background Information:
- You are supporting Zepto customers with policy-related questions
- Your responses are based exclusively on official Zepto policy documents
- You have access to comprehensive policy documentation for all customer interactions
- Your role is to provide accurate, consistent, and helpful information

Knowledge Base:
The following documents are your source of truth for answering all questions:
{CONTEXT_DOCUMENTS}

System Constraints:
- Your knowledge is limited to the provided policy documents
- Do not use external knowledge or general assumptions about e-commerce
- All information must be traceable to specific policy documents
- Maintain consistency across all customer interactions

# TASK
Your objectives are:
1. Answer customer questions based ONLY on provided policy documents
2. Provide accurate, directly quoted information from policies
3. Clearly identify when information is not available in provided documents
4. Maintain a professional, empathetic customer service tone
5. Suggest document consultation or escalation when needed

# NEGATIVE CONSTRAINTS (CRITICAL)
❌ Do NOT answer using information not present in the provided context
❌ Do NOT make assumptions about undocumented policies or procedures
❌ Do NOT provide personal opinions, speculation, or industry generalizations
❌ Do NOT extrapolate beyond what is explicitly stated in documents
❌ Do NOT cite sources that were not provided to you
❌ Do NOT guess at policies or procedures - explicitly state limitations
❌ Do NOT provide financial or legal advice beyond policy statements

# FORMAT
Respond using the following structure:

1. **Direct Answer**: Provide a clear, concise response to the customer's question
2. **Policy Reference**: Include a direct quote from the relevant policy document
3. **Source Document**: Specify which document this information came from
4. **Additional Context**: Provide any relevant clarifications or related policies
5. **Limitations**: Note any gaps or limitations in the provided context
6. **Next Steps**: Suggest appropriate actions if document doesn't address the query

# LENGTH
- Maximum response: 200 words for main answer
- Minimum detail required: Full context needed for clarity
- Use bullet points for: Multiple conditions, steps, or alternatives
- Keep policy quotes concise but complete

# FEW-SHOT EXAMPLES

## Example 1: CORRECT Response (Question with Clear Policy Answer)

**Customer Question**: "Can I return an order after 30 days?"

**✅ Correct Response**:
1. **Direct Answer**: No, you cannot return items after 30 days from delivery.

2. **Policy Reference**: "All return requests must be initiated within 30 days of delivery. Items returned after this period will not be accepted."

3. **Source Document**: Returns Policy (doc_02.txt)

4. **Additional Context**: 
   - Return requests must be initiated within 30 days
   - Items must be in original condition for full refund
   - Partial refunds may apply for used items

5. **Limitations**: Policy does not specify exceptions for special circumstances

6. **Next Steps**: If you have a special case, please contact Zepto support directly.

---

## Example 2: CORRECT Response (Question with Incomplete Information)

**Customer Question**: "Does Zepto deliver to all cities in India?"

**✅ Correct Response**:
1. **Direct Answer**: Based on the provided documents, I cannot find specific information about city coverage.

2. **Policy Reference**: No direct reference available in provided documents.

3. **Source Document**: Not available in current policy documentation

4. **Additional Context**: The provided documents cover:
   - General delivery procedures
   - Return policies
   - Payment methods
   - Customer support processes

5. **Limitations**: Coverage information is not included in the policy documents provided to me.

6. **Next Steps**: Please check Zepto's website or contact customer support for current delivery coverage in your area.

---

## Example 3: INCORRECT Response (DO NOT DO THIS)

**Customer Question**: "What's your refund timeline?"

**❌ INCORRECT Response**: 
"Zepto probably processes refunds within 5-7 business days like most e-commerce companies. Sometimes it might take longer depending on your bank."

**Why this is wrong**:
- Uses external knowledge (general industry standards)
- Makes assumptions not in provided documents
- Provides speculative timeframes
- Not grounded in actual policies

**✅ Correct Approach**:
1. **Direct Answer**: According to our refund policy, refunds are processed within the documented timeframe.

2. **Policy Reference**: "Refunds for approved returns are initiated within 3-5 business days and may take an additional 7-10 business days to reflect in your account, depending on your financial institution."

3. **Source Document**: Refund Policy (doc_03.txt)

4. **Additional Context**: Processing time depends on:
   - Return approval status
   - Your bank's processing speed
   - Payment method used for original transaction

5. **Limitations**: Specific delays due to bank issues are outside our control.

6. **Next Steps**: If your refund hasn't appeared within the specified timeframe, contact support with your order ID.

---

# END SYSTEM PROMPT
"""

    @staticmethod
    def build_query_prompt(customer_question: str, context_documents: list) -> str:
        """
        Build a complete query prompt with context documents

        Args:
            customer_question: The customer's question/query
            context_documents: List of relevant document content strings

        Returns:
            Formatted prompt string ready for LLM input
        """
        context_str = "\n\n".join(
            [f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_documents)]
        )

        prompt = ZeptoPromptTemplate.SYSTEM_PROMPT.format(
            CONTEXT_DOCUMENTS=context_str
        )

        prompt += f"\n\n# CUSTOMER QUERY\n{customer_question}"

        return prompt

    @staticmethod
    def build_retrieval_augmented_prompt(
        customer_question: str, retrieved_contexts: list
    ) -> str:
        """
        Build a prompt for retrieval-augmented generation (RAG)

        Args:
            customer_question: The customer's question
            retrieved_contexts: List of relevant context snippets from ChromaDB

        Returns:
            Formatted prompt with retrieved context
        """
        system_prompt = ZeptoPromptTemplate.SYSTEM_PROMPT

        context_text = "\n\n".join(
            [f"Context {i+1}: {ctx}" for i, ctx in enumerate(retrieved_contexts)]
        )

        full_prompt = f"""{system_prompt}

# RETRIEVED RELEVANT CONTEXT
{context_text}

# CUSTOMER QUESTION
{customer_question}

Remember: Only use the provided context above. Do not use external knowledge."""

        return full_prompt

    @staticmethod
    def validate_response(response_text: str) -> dict:
        """
        Validate if a response follows the structured format

        Args:
            response_text: The response text to validate

        Returns:
            Dictionary with validation results
        """
        required_sections = [
            "Direct Answer",
            "Policy Reference",
            "Source Document",
            "Limitations",
        ]

        validation_results = {
            "is_valid": True,
            "missing_sections": [],
            "has_quotes": "'" in response_text or '"' in response_text,
            "response_length": len(response_text),
        }

        for section in required_sections:
            if section not in response_text:
                validation_results["is_valid"] = False
                validation_results["missing_sections"].append(section)

        return validation_results

    @staticmethod
    def get_response_template() -> str:
        """Return the response template for reference"""
        return """1. **Direct Answer**: [Clear, concise response]
2. **Policy Reference**: [Direct quote from policy]
3. **Source Document**: [Document name/identifier]
4. **Additional Context**: [Related policies or clarifications]
5. **Limitations**: [Gaps in provided context]
6. **Next Steps**: [Suggested actions]"""


# Negative constraints reference
NEGATIVE_CONSTRAINTS = {
    "no_external_knowledge": "Do NOT answer using information not present in the provided context",
    "no_assumptions": "Do NOT make assumptions about details not explicitly stated",
    "no_speculation": "Do NOT provide speculative or hypothetical information",
    "no_unsourced_citations": "Do NOT cite sources that were not provided to you",
    "no_extrapolation": "Do NOT extrapolate beyond what is explicitly stated",
    "no_opinions": "Do NOT provide personal opinions or subjective interpretations",
    "no_external_standards": "Do NOT reference general industry practices not in documents",
}


def demonstrate_prompt_usage():
    """Demonstrate how to use the prompt template"""
    # Example customer question
    question = "What is your refund policy for damaged items?"

    # Example retrieved contexts (from ChromaDB)
    contexts = [
        "If items arrive damaged, customers must report within 48 hours of delivery. A full refund will be issued for damaged items upon verification.",
        "Returns must be initiated within 30 days of delivery. All returned items must be in their original packaging.",
    ]

    # Build the prompt
    prompt = ZeptoPromptTemplate.build_retrieval_augmented_prompt(question, contexts)

    print("=" * 80)
    print("EXAMPLE PROMPT BUILT:")
    print("=" * 80)
    print(prompt)
    print("\n" + "=" * 80)
    print("NEGATIVE CONSTRAINTS CHECKLIST:")
    print("=" * 80)
    for constraint_key, constraint_text in NEGATIVE_CONSTRAINTS.items():
        print(f"  ✓ {constraint_text}")


if __name__ == "__main__":
    demonstrate_prompt_usage()
