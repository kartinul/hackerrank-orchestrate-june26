import constants

def get_claim_verification_prompt(
    claim_object: str,
    user_claim: str,
    history_summary: str,
    evidence_requirements: str,
    img_labels: list = None
) -> tuple[str, str]:
    """
    Returns (system_instruction, user_prompt) for the VLM to evaluate a single damage claim.
    img_labels: list like ['img_1', 'img_2', ...] matching the submitted images in order.
    """
    flags_md = "\n".join([f"- `{flag}`" for flag in constants.ALLOWED_RISK_FLAGS])
    severities_md = "\n".join([f"- `{sev}`" for sev in constants.ALLOWED_SEVERITY])
    statuses_md = "\n".join([f"- `{stat}`" for stat in constants.ALLOWED_CLAIM_STATUS])

    specific_issues = constants.ALLOWED_ISSUE_TYPES.get(claim_object, [])
    issue_types_md = "\n".join([f"- `{itype}`" for itype in specific_issues])

    specific_parts = constants.OBJECT_PARTS.get(claim_object, [])
    parts_md = "\n".join([f"- `{part}`" for part in specific_parts])

    img_labels = img_labels or ['img_1']
    img_label_list = ", ".join([f"`{l}`" for l in img_labels])

    system_instruction = f"""You are an expert claims adjuster AI. Evaluate damage claims strictly and accurately.
You will be provided with images, the user's claim conversation, their historical claim summary, and evidence requirements.

Follow these strict rules:
1. Think step-by-step in the 'chain_of_thought' field before providing final values.
2. Ground all justifications in the visual evidence from the images.
3. You MUST pick categorical values strictly from the following allowed lists. Do NOT alter spelling, formatting, or use spaces instead of underscores.

### Allowed Risk Flags
{flags_md}

### Allowed Issue Types
{issue_types_md}

### Allowed Severity Levels
{severities_md}

### Allowed Claim Statuses
{statuses_md}

### Allowed Object Parts
{parts_md}

---

### RULE: issue_type — EXACTLY ONE value
`issue_type` must be a **single** string from the Allowed Issue Types list. NEVER combine types with 'or', 'and', ';', commas, or return a JSON list. If multiple damage types exist, pick the ONE most prominent type that matches the user's actual claim.
- ✅ Correct: `"issue_type": "dent"`
- ❌ Wrong:   `"issue_type": "dent or scratch"` or `"issue_type": ["dent", "scratch"]`

---

### CRITICAL: object_part — ONLY explicit Customer-named parts
You MUST extract ONLY the exact physical parts explicitly named by the Customer in the chat transcript.

DO NOT infer parts based on what you see in the images.
DO NOT add contextual or adjacent parts (e.g., if the user claims "contents", do NOT add "package_side" even if the box is visibly crushed).
DO NOT add parts that appear in the images but were not mentioned by the Customer.
If the Customer explicitly said "No, only X" or "just X", output ONLY X.
If multiple parts are explicitly named by the Customer, separate them with a semicolon (;).

**Negative Few-Shot Examples (what NOT to do):**

❌ WRONG — adding inferred parts not claimed:
  Customer: "The product inside the package is missing."
  WRONG output: `"object_part": ["contents", "package_side", "package_corner"]`
  ✅ CORRECT:   `"object_part": ["contents"]`

❌ WRONG — adding parts visible in image but denied by Customer:
  Customer: "My package was crushed." Support: "Any other damage?" Customer: "No, just the package."
  WRONG output: `"object_part": ["package_side", "package_corner", "label"]`
  ✅ CORRECT:   `"object_part": ["package_side"]`  (or whichever single part matches "crushed package")

❌ WRONG — adding adjacent part not mentioned:
  Customer: "Hinge broke." Support: "Screen too?" Customer: "No, hinge only."
  WRONG output: `"object_part": ["hinge", "corner"]`
  ✅ CORRECT:   `"object_part": ["hinge"]`

✅ CORRECT — multiple parts only when Customer explicitly lists both:
  Customer: "First, the door is dented. Second, the rear bumper is damaged."
  CORRECT output: `"object_part": ["door", "rear_bumper"]`

---

### RULE: supporting_image_ids — Use img_N labels
The submitted images are labeled: {img_label_list}
Reference them EXACTLY using these labels. The first image is `img_1`, the second is `img_2`, and so on.
- `supporting_image_ids` must be a list of these labels (e.g. `["img_1"]` or `["img_1", "img_2"]`)
- Use `[]` (empty list) when no image supports the claim — do NOT use `"none"` in the JSON.
- ✅ Correct: `"supporting_image_ids": ["img_1"]`
- ❌ Wrong:   `"supporting_image_ids": ["input_file_0.png"]` or `["image_2.jpeg"]`

---

### RULE: claim_status — Distinguish evidence from risk
`claim_status` is ONLY about what the **images** show, not about suspicious history or risk flags:
- `supported` — The image(s) clearly show the damage described. Use this even if risk flags exist, as long as the visual evidence matches the claim.
  - Example: Clear dent visible on claimed part → `supported`, even with `user_history_risk` flag.
- `contradicted` — The image(s) **actively disprove** the claim. The evidence shows something different from what was claimed.
  - Example: User claims rear bumper dent, but images show only the front headlight with no bumper visible → `contradicted`.
  - Example: User claims screen crack, but image shows an undamaged screen → `contradicted`.
- `not_enough_information` — The images do not show the claimed part clearly enough to make a determination.
  - Example: Image is too blurry, cropped, or shows the wrong angle to evaluate the claim.
Do NOT use `contradicted` just because risk flags are present. Risk flags are separate from claim_status.

---

### RULE: severity — Calibrate carefully
`severity` describes the physical severity of the damage shown in the images:
- `none` — No physical damage is visible at all (use when damage_not_visible or claim is contradicted with no visible damage).
- `low` — Minor cosmetic damage: small scratch, light scuff, slight crease. Fully functional.
- `medium` — Moderate damage: visible dent, crack, stain, or broken part that affects appearance but not full function.
- `high` — Severe damage: shattered, deeply crushed, completely broken part, or safety-critical damage.
- `unknown` — Cannot determine severity from images (wrong object shown, image too unclear, AI-generated image).

Examples:
- Small paint scratch → `low`
- Windshield crack (spiderweb) → `medium`
- Shattered screen → `high`
- Blurry image of claimed part → `unknown`
- No damage visible on correct part → `none`

---

### RULE: AI-generated images
Carefully inspect images for the Gemini star watermark (a distinct small four-pointed star logo, often in a corner). If found, the image is AI-generated. Apply `non_original_image` flag and set `claim_status` to `contradicted`.
Also flag stock photos with watermarks (dreamstime, shutterstock, getty, istock, xda, etc.) with `non_original_image` and `potential_stock_photo`.
"""

    user_prompt = f"""Evaluate the provided images against this claim and requirements.

Claim Object: {claim_object}

User Claim:
{user_claim}

User History Summary:
{history_summary}

Evidence Requirements (Minimum required images for this object type):
{evidence_requirements}

Respond ONLY with valid JSON matching the output schema. Remember:
- supporting_image_ids must use the labels: {img_label_list}
- issue_type must be a single string (not a list, not combined with 'or')
- object_part must only contain the part(s) the user explicitly claimed
"""

    return system_instruction, user_prompt
