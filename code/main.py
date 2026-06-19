import os
import time
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from common_helpers import read_dataset, write_dataset
from orchestrator import process_claim_with_ai
from ai_helpers import get_api_keys

if __name__ == "__main__":
    # Load datasets
    claims_df = read_dataset("../dataset/claims.csv")
    user_history_df = read_dataset("../dataset/user_history.csv")
    evidence_req_df = read_dataset("../dataset/evidence_requirements.csv")
    
    results = []

    # Rate limit setup: 5 requests per key per minute
    num_keys = len(get_api_keys())
    batch_size = 5 * num_keys if num_keys > 0 else 5
    processed_in_batch = 0
    batch_start_time = time.time()

    # Process each claim
    for index, row in claims_df[0:2].iterrows():
        # Rate limiting block
        if processed_in_batch >= batch_size:
            elapsed = time.time() - batch_start_time
            if elapsed < 60:
                sleep_time = 60 - elapsed + 1
                print(f"Rate limit batch size ({batch_size}) reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            
            # Reset batch trackers
            processed_in_batch = 0
            batch_start_time = time.time()

        # Make a single holistic AI call per claim
        output_row = process_claim_with_ai(row, evidence_req_df, user_history_df)
        processed_in_batch += 1
        
        # process_claim_with_ai returns a dictionary mapping to the output schema.
        # Fallback to an empty schema if the API fails entirely.
        if output_row:
            results.append(output_row)
        else:
            print(f"Warning: AI process failed for user_id {row.get('user_id')}. Appending fallback.")
            results.append({
                'user_id': row.get('user_id'),
                'image_paths': row.get('image_paths'),
                'user_claim': row.get('user_claim'),
                'claim_object': row.get('claim_object'),
                'evidence_standard_met': False,
                'evidence_standard_met_reason': 'Failed to process',
                'risk_flags': 'none',
                'issue_type': 'unknown',
                'object_part': 'unknown',
                'claim_status': 'not_enough_information',
                'claim_status_justification': 'Failed to process',
                'supporting_image_ids': 'none',
                'valid_image': True,
                'severity': 'unknown'
            })
    
    # Write output
    results_df = pd.DataFrame(results)
    write_dataset("../dataset/output.csv", results_df)
    print(f"Processed {len(results_df)} claims and wrote to ../dataset/output.csv")
