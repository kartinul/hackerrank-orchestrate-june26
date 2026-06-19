ALLOWED_CLAIM_STATUS = [
    "supported",
    "contradicted",
    "not_enough_information"
]

ALLOWED_ISSUE_TYPES = {
    "car": ["dent", "scratch", "crack", "broken_part", "none", "unknown"],
    "laptop": ["crack", "broken_part", "stain", "dent", "none", "unknown"],
    "package": ["crushed_packaging", "torn_packaging", "water_damage", "stain", "none", "unknown"]
}

ALLOWED_SEVERITY = [
    "none",
    "low",
    "medium",
    "high",
    "unknown"
]

ALLOWED_RISK_FLAGS = [
    "none",
    "claim_mismatch",
    "user_history_risk",
    "manual_review_required",
    "wrong_angle",
    "damage_not_visible",
    "blurry_image",
    "non_original_image",
    "cropped_or_obstructed",
    "wrong_object",
    "text_instruction_present",
    "mismatched_vehicle_parts",
    "inconsistent_damage_between_views",
    "potential_stock_photo",
    "history_of_exaggerated_claims",
    "damage_severity_inconsistent_with_claim_description"
]

OBJECT_PARTS = {
    "car": ["rear_bumper", "front_bumper", "windshield", "side_mirror", "headlight", "taillight", "door", "door_panel", "hood", "left_headlight", "unknown"],
    "laptop": ["screen", "hinge", "keyboard", "corner", "trackpad", "lid", "unknown"],
    "package": ["package_corner", "seal", "package_side", "contents", "label", "unknown"]
}

COLUMN_ORDER = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity"
]
