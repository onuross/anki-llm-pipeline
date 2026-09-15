import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==============================================================================
# 1. SETUP & PATHS
# ==============================================================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("CRITICAL: GEMINI_API_KEY is not set in the .env file.")

BASE_DIR = Path(__file__).parent.parent.resolve()
TXT_DIR = BASE_DIR / "data" / "txt"

INPUT_FILE = TXT_DIR / "phase1_raw_words.txt"
OUTPUT_FILE = TXT_DIR / "phase2_normalized_words.txt"

BATCH_SIZE = 20
MODEL_ID = "gemini-3.1-flash-lite-preview"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

SYSTEM_PROMPT = """
Act as a strict German grammar normalizer and data cleaning API.
I will provide you with a JSON array of raw German words/phrases.

Your task is to CLEAN and NORMALIZE them according to these STRICT rules:

1. GARBAGE ELIMINATION (Drop these entirely, do not include in output):
   - Proper nouns (Names, Cities, Countries, Brands like Alexander, Goethe, München).
   - Personal pronouns (ich, du, er, sie, es, wir, ihr, sie, Ihnen).
   - Basic conjunctions and prepositions (und, oder, mit, zu, von).
   - Numbers, symbols, page numbers, gibberish (e.g., "xyz").
   - CRITICAL EXCEPTION: Do NOT drop valid everyday nouns, verbs, or adjectives just because they are simple (e.g., Haus, Katze, machen). Keep them!

2. NORMALIZATION PROTOCOL (For valid words):
   - NOUNS: Convert to singular. Capitalize the first letter. STRICT RULE: DO NOT add articles (der/die/das). DO NOT add plural markers. (Example: "Häuser" -> "Haus").
   - VERBS: Convert to bare infinitive. Join separable verbs. (Example: "kaufte ein" -> "einkaufen", "wirfst" -> "werfen").
   - ADJECTIVES/ADVERBS: Convert to base positive form. (Example: "am schnellsten" -> "schnell", "besser" -> "gut").

OUTPUT FORMAT:
Return ONLY a valid JSON array of strings containing the normalized words.
If all words in the input batch are garbage, return an empty array: []
Do not add any markdown formatting or conversational text outside the JSON array.
"""


def run_normalization_gate():
    if not INPUT_FILE.exists():
        logging.error(f"❌ Error: '{INPUT_FILE.name}' not found! Run Phase 1 first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pending_words = [line.strip() for line in f.readlines() if line.strip()]

    total_words = len(pending_words)
    if total_words == 0:
        logging.info("✅ Input file is empty. All words have been normalized!")
        return

    logging.info(
        f"🚀 Phase 2: LLM Normalization Gate Opened. Pending Words: {total_words}"
    )
    client = genai.Client(api_key=API_KEY)

    total_processed = 0
    total_requests = (total_words // BATCH_SIZE) + 1

    while pending_words:
        current_batch = pending_words[:BATCH_SIZE]
        current_request = (total_processed // BATCH_SIZE) + 1

        max_retries = 5
        success = False

        for attempt in range(max_retries):
            try:
                logging.info(
                    f"📦 Request {current_request}/{total_requests} | Processing words {total_processed} to {total_processed + len(current_batch)}..."
                )

                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=f"{SYSTEM_PROMPT}\n\nINPUT BATCH:\n{json.dumps(current_batch)}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )

                clean_batch = json.loads(response.text)

                if clean_batch:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write("\n".join(clean_batch) + "\n")

                # Stateful update
                pending_words = pending_words[BATCH_SIZE:]
                with open(INPUT_FILE, "w", encoding="utf-8") as f:
                    f.write("\n".join(pending_words))

                total_processed += len(current_batch)
                logging.info(
                    f"   ✅ {len(clean_batch)} words passed the gate. Remaining: {len(pending_words)}"
                )

                success = True
                time.sleep(2)  # API breather
                break

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg:
                    wait_time = 15 * (attempt + 1)
                    logging.warning(
                        f"   ⏳ [429 Error] Cooling down... Waiting {wait_time}s."
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(f"   ❌ JSON/Connection Error: {e}. Retrying in 5s.")
                    time.sleep(5)

        if not success:
            logging.critical(
                "\n🛑 Critical error: Batch failed after 5 attempts. Halting system."
            )
            logging.info(
                "Input file state preserved. You can restart with 0 data loss."
            )
            break

    if not pending_words:
        logging.info(f"\n🎉 --- PHASE 2 COMPLETED ---")
        logging.info(f"📁 All words normalized and saved to '{OUTPUT_FILE.name}'.")
        logging.info("Next step: Phase 3 (Python deduplication).")


if __name__ == "__main__":
    run_normalization_gate()
