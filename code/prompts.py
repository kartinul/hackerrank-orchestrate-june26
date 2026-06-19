def get_claim_verification_prompt(claim_object: str, user_claim: str, history_summary: str, evidence_requirements: str) -> str:
    """
    Returns the system prompt for the VLM to evaluate a single damage claim.
    """
    return f"""You are an expert claims adjuster. You must evaluate a damage claim for a {claim_object}.

User Claim:
{user_claim}

User History Summary:
{history_summary}

Evidence Requirements (Minimum required images for this object type):
{evidence_requirements}

Evaluate the provided images against the claim and requirements.
Return a strictly formatted JSON object matching the required schema:
{{
    "evidence_standard_met": bool,
    "evidence_standard_met_reason": "string",
    "risk_flags": "string (semicolon separated) or 'none'",
    "issue_type": "string",
    "object_part": "string",
    "claim_status": "supported|contradicted|not_enough_information",
    "claim_status_justification": "string",
    "supporting_image_ids": "string (semicolon separated) or 'none'",
    "valid_image": bool,
    "severity": "none|low|medium|high|unknown"
}}
"""
