import os
import json
import time
import logging
import fitz  # PyMuPDF
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
PDF_DIR = BASE_DIR / "data" / "pdf"
TXT_DIR = BASE_DIR / "data" / "txt"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = TXT_DIR / "phase1_raw_words.txt"
PDF_FILES = ["Goethe-Zertifikat_B1_Wortliste.pdf"]
MODEL_ID = "gemini-3.1-flash-lite-preview"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

SYSTEM_PROMPT = """
Act as an expert German linguist and data parser. I will provide you with raw text extracted from a single page of an all-German vocabulary book.

Context & Challenge: The text contains headwords, their German explanations, and German example sentences. Your task is to extract ONLY the main target B1 level vocabulary words being taught. Do NOT blindly extract every complex word from the explanations or example sentences, which would cause duplication and context loss.

Strict Extraction Rules:
1. Identify the Headword (Naked Lemma): Focus on extracting the ABSOLUTE BASE FORM of the target words.
2. Normalize (CRITICAL):
    - Nouns MUST be extracted WITHOUT definite articles. Example: output 'Einkommen' (NOT 'das Einkommen').
    - Verbs MUST be in the bare infinitive form. Remove reflexive pronouns ('sich') and prepositions. Example: output 'freuen' (NOT 'sich freuen auf'). Separable verbs must be joined (e.g., 'ausreichen').
    - Adjectives/Adverbs MUST be in their base positive form.
3. Deduplicate: Never output the same word twice from this page.
4. Exclude: Ignore A1/A2 basic words (ich, du, machen), proper nouns, numbers, and symbols.

Output Format:
Return ONLY a raw JSON array of strings. No markdown formatting, no conversational text, no explanations.
Example: ["Einkommen", "bestätigen", "ausreichen", "freuen"]
"""


def run_pdf_scraper():
    client = genai.Client(api_key=API_KEY)
    unique_words = set()

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            old_words = [line.strip() for line in f.readlines() if line.strip()]
            unique_words.update(old_words)
            logging.info(
                f"🔄 Recovered {len(unique_words)} words from previous session.\n"
            )

    logging.info("🚀 Phase 1: PDF Extractor Started...\n")

    for pdf_name in PDF_FILES:
        pdf_path = PDF_DIR / pdf_name
        if not pdf_path.exists():
            logging.error(f"❌ Error: {pdf_name} not found in {PDF_DIR}!")
            continue

        logging.info(f"\n📂 Processing: {pdf_name}")
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            for page_num in range(total_pages):
                text = pdf_document[page_num].get_text("text").strip()
                if len(text) < 50:
                    continue

                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=f"{SYSTEM_PROMPT}\n\n--- PAGE TEXT ---\n{text}",
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json"
                            ),
                        )

                        page_words = json.loads(response.text)
                        previous_count = len(unique_words)
                        unique_words.update(page_words)
                        new_additions = len(unique_words) - previous_count

                        # Stateful save
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            f.write("\n".join(sorted(unique_words)))

                        logging.info(
                            f"   ✅ Page {page_num + 1}/{total_pages} | +{new_additions} new words (Total: {len(unique_words)})"
                        )
                        time.sleep(4)
                        break

                    except Exception as e:
                        error_msg = str(e).lower()
                        if (
                            "429" in error_msg
                            or "quota" in error_msg
                            or "exhausted" in error_msg
                        ):
                            wait_time = 15 * (attempt + 1)
                            logging.warning(
                                f"   ⏳ [429 Error] Cooling down engines... Waiting {wait_time}s. (Attempt {attempt+1}/{max_retries})"
                            )
                            time.sleep(wait_time)
                        else:
                            logging.error(
                                f"   ❌ Page {page_num + 1} parsing failed: {e}"
                            )
                            break

        except Exception as e:
            logging.error(f"❌ Could not read PDF ({pdf_name}): {e}")

    logging.info("\n🎉 --- PHASE 1 COMPLETED ---")
    logging.info(f"🔥 Total Extracted Raw Words: {len(unique_words)}")
    logging.info(f"📁 Saved to: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    run_pdf_scraper()
