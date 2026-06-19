import re
import pandas as pd
from typing import Dict, Any
from ai_helpers import call_vlm, parse_json_response
from prompts import get_claim_verification_prompt


def _normalize_img_id(val: str, img_labels: list) -> str:
    """Map whatever the model called the image back to img_N (1-based).

    The VLM tool receives images with internal names like:
      input_file_0, input_file_1  (0-based, tool-generated)
      image_0.png, img_0.jpg      (0-based)
      image_1.jpeg, img_1         (1-based, our labels)
    We always return from img_labels (img_1, img_2, ...).
    """
    val = val.strip()
    # Already in correct format
    if val in img_labels:
        return val
    # Extract trailing number
    nums = re.findall(r'\d+', val)
    if nums:
        n = int(nums[-1])
        # Determine if this is 0-based or 1-based:
        # tool-generated names (input_file_N, image_N, img_N.ext) are 0-based
        # our own labels (img_1, img_2) are 1-based and already handled above
        is_zero_based = 'input_file' in val or re.search(r'image[_\s]*\d', val) or re.search(r'^img[_\s]*\d', val)
        if is_zero_based:
            idx = n  # 0-based index directly
        else:
            idx = n - 1  # 1-based -> convert to 0-based
        if 0 <= idx < len(img_labels):
            return img_labels[idx]
    # Fallback: first image
    return img_labels[0] if img_labels else 'none'


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

    # Build img_N labels (img_1, img_2, ...) matching position in image_paths
    img_labels = [f"img_{i+1}" for i in range(len(image_paths))]

    # Get user history context
    history_row = user_history_df[user_history_df['user_id'] == user_id]
    if not history_row.empty:
        hr = history_row.iloc[0]
        history_summary = (
            f"Total Past Claims: {hr['past_claim_count']}\n"
            f"Accepted Claims: {hr['accept_claim']}\n"
            f"Rejected Claims: {hr['rejected_claim']}\n"
            f"Manual Review Claims: {hr['manual_review_claim']}\n"
            f"Claims in Last 90 Days: {hr['last_90_days_claim_count']}\n"
            f"History Flags: {hr['history_flags']}\n"
            f"Summary Notes: {hr['history_summary']}"
        )
    else:
        history_summary = "No history available for this user."

    # Get evidence requirements for this object
    reqs = evidence_req_df[evidence_req_df['claim_object'].isin([claim_object, 'all'])]
    reqs_text = reqs[['applies_to', 'minimum_image_evidence']].to_string(index=False)

    # Construct the prompt (pass img_labels so prompt can reference them)
    system_instruction, prompt = get_claim_verification_prompt(
        claim_object, user_claim, history_summary, reqs_text, img_labels
    )

    # Make the single VLM call
    raw_response = call_vlm(system_instruction, prompt, image_paths)

    # Parse the response
    parsed_json = parse_json_response(raw_response)

    if parsed_json:
        # --- issue_type: always a single clean value ---
        raw_issue = parsed_json.get('issue_type', 'unknown')
        if isinstance(raw_issue, list):
            raw_issue = raw_issue[0] if raw_issue else 'unknown'
        # Strip composites like "dent or scratch" or "dent;scratch" -> take first token
        raw_issue = re.split(r'\s+or\s+|\s+and\s+|;', str(raw_issue))[0].strip()
        issue_type = raw_issue if raw_issue else 'unknown'

        # --- supporting_image_ids: normalize to img_N labels ---
        raw_ids = parsed_json.get('supporting_image_ids', [])
        if isinstance(raw_ids, str):
            if raw_ids.lower() in ('none', ''):
                raw_ids = []
            else:
                raw_ids = [s.strip() for s in raw_ids.split(';') if s.strip()]

        if raw_ids:
            normalized = [_normalize_img_id(v, img_labels) for v in raw_ids]
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for v in normalized:
                if v not in seen:
                    seen.add(v)
                    deduped.append(v)
            supporting_image_ids = ';'.join(deduped)
        else:
            supporting_image_ids = 'none'

        # Construct the final row schema incorporating the known IDs
        return {
            'user_id': user_id,
            'image_paths': row['image_paths'],
            'user_claim': user_claim,
            'claim_object': claim_object,
            'evidence_standard_met': parsed_json.get('evidence_standard_met', False),
            'evidence_standard_met_reason': parsed_json.get('evidence_standard_met_reason', 'Unknown'),
            'risk_flags': ';'.join(parsed_json.get('risk_flags', [])) if parsed_json.get('risk_flags') else 'none',
            'issue_type': issue_type,
            'object_part': ';'.join(parsed_json.get('object_part', [])) if isinstance(parsed_json.get('object_part'), list) else parsed_json.get('object_part', 'unknown'),
            'claim_status': parsed_json.get('claim_status', 'not_enough_information'),
            'claim_status_justification': parsed_json.get('claim_status_justification', 'Failed to generate justification'),
            'supporting_image_ids': supporting_image_ids,
            'valid_image': parsed_json.get('valid_image', True),
            'severity': parsed_json.get('severity', 'unknown')
        }

    return None
