import pandas as pd
from typing import Dict, Any
from ai_helpers import call_vlm, parse_json_response
from prompts import get_claim_verification_prompt

def process_claim_with_ai(row: pd.Series, evidence_req_df: pd.DataFrame, user_history_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a single claim using one holistic AI call.
    Gathers all context (images, user claim, user history, evidence requirements), 
    makes a single call to a Vision-Language Model, and parses the structured JSON response 
    to return a dictionary mapping to the required output schema.
    """
    user_id = row['user_id']
    claim_object = row['claim_object']
    image_paths = row['image_paths'].split(';')
    user_claim = row['user_claim']
    
    # Get user history context
    history_row = user_history_df[user_history_df['user_id'] == user_id]
    history_summary = history_row.iloc[0]['history_summary'] if not history_row.empty else "No history available"
    
    # Get evidence requirements for this object
    reqs = evidence_req_df[evidence_req_df['claim_object'].isin([claim_object, 'all'])]
    reqs_text = reqs[['applies_to', 'minimum_image_evidence']].to_string(index=False)
    
    # Construct the prompt
    prompt = get_claim_verification_prompt(claim_object, user_claim, history_summary, reqs_text)
    
    # Make the single VLM call
    raw_response = call_vlm(prompt, image_paths)
    
    # Parse the response
    parsed_json = parse_json_response(raw_response)
    
    if parsed_json:
        # Construct the final row schema incorporating the known IDs
        return {
            'user_id': user_id,
            'image_paths': row['image_paths'],
            'user_claim': user_claim,
            'claim_object': claim_object,
            'evidence_standard_met': parsed_json.get('evidence_standard_met', False),
            'evidence_standard_met_reason': parsed_json.get('evidence_standard_met_reason', 'Unknown'),
            'risk_flags': parsed_json.get('risk_flags', 'none'),
            'issue_type': parsed_json.get('issue_type', 'unknown'),
            'object_part': parsed_json.get('object_part', 'unknown'),
            'claim_status': parsed_json.get('claim_status', 'not_enough_information'),
            'claim_status_justification': parsed_json.get('claim_status_justification', 'Failed to generate justification'),
            'supporting_image_ids': parsed_json.get('supporting_image_ids', 'none'),
            'valid_image': parsed_json.get('valid_image', True),
            'severity': parsed_json.get('severity', 'unknown')
        }
    
    return None
