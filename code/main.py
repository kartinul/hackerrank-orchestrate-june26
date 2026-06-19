import os
import time
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from common_helpers import read_dataset, write_dataset
from orchestrator import process_claim_with_ai
from ai_helpers import get_api_keys

if __name__ == "__main__":
    ENTRIES = -1 # Set to -1 to process all, or a specific number (e.g. 5) to process the first N claims
    
    # Load datasets
    claims_df = read_dataset("../dataset/claims.csv")
    user_history_df = read_dataset("../dataset/user_history.csv")
    evidence_req_df = read_dataset("../dataset/evidence_requirements.csv")
    
    results = []

    num_keys = len(get_api_keys())
    max_workers = 5 * num_keys if num_keys > 0 else 5

    claims_to_process = claims_df if ENTRIES == -1 else claims_df.head(ENTRIES)
    actual_workers = min(max_workers, len(claims_to_process))

    print(f"\nStarting {actual_workers} parallel agents...\n")

    def process_single_claim(row):
        try:
            output_row = process_claim_with_ai(row, evidence_req_df, user_history_df)
            if output_row:
                return output_row
            else:
                print(f"Warning: AI process failed for user_id {row.get('user_id')}. Appending fallback.")
                return {
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
                }
        except Exception as e:
            print(f"Exception for user_id {row.get('user_id')}: {e}")
            return {
                'user_id': row.get('user_id'),
                'image_paths': row.get('image_paths'),
                'user_claim': row.get('user_claim'),
                'claim_object': row.get('claim_object'),
                'evidence_standard_met': False,
                'evidence_standard_met_reason': f'Exception: {e}',
                'risk_flags': 'none',
                'issue_type': 'unknown',
                'object_part': 'unknown',
                'claim_status': 'not_enough_information',
                'claim_status_justification': f'Exception: {e}',
                'supporting_image_ids': 'none',
                'valid_image': True,
                'severity': 'unknown'
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_claim, row): index for index, row in claims_to_process.iterrows()}
        
        results_dict = {}
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Claims"):
            original_index = futures[future]
            results_dict[original_index] = future.result()
            
    # Reconstruct results in original input order
    results = [results_dict[i] for i in sorted(results_dict.keys())]
            
    # Write output
    results_df = pd.DataFrame(results)
    write_dataset("../dataset/output.csv", results_df)
    print(f"Processed {len(results_df)} claims and wrote to ../dataset/output.csv")
