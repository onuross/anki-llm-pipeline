import os
import json
import re
import csv
import socket
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# ==============================================================================
# 0. INITIALIZATION & SECURITY (.env)
# ==============================================================================
# Load environment variables from .env file (Prevents hardcoding API keys)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("CRITICAL: GEMINI_API_KEY is not set in the .env file.")

MODEL_ID = "gemini-3.6-flash"

# ==============================================================================
# 1. DYNAMIC PATHS (Portability)
# ==============================================================================
# Dynamically define base directories so the script runs on any machine
BASE_DIR = Path(
    __file__
).parent.parent.resolve()  # Adjust based on your final folder structure
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
LOG_FILE = BASE_DIR / "ankibot.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# ==============================================================================
# 2. CONFIGURATIONS (i18n Supported)
# ==============================================================================

GERMAN_CONFIG = {
    "language": "German",
    "base_language": "Turkish",  # i18n: Can be changed to "Spanish", "English", etc.
    "txt_path": DATA_DIR / "txt" / "de_example.txt",
    "csv_path": DATA_DIR / "csv" / "de_example.csv",
    "main_csv_path": DATA_DIR / "csv" / "DEUTSCH.csv",
    "manual_prompt_path": PROMPTS_DIR / "de_3_pro_qa.txt",
    "pro_prompt_path": PROMPTS_DIR / "de_4_pro_prod.txt",
    "deck_name": "Deutsch",
    "note_type": "German_Cloze",
    "field_sentence": "Sentence_German",
    "limit": 6,
    "batch_size": 2,
    "pro_limit": 15,  # Fallback: Max words for manual Pro model generation
    "voice": "de-DE-ConradNeural",
}

ENGLISH_CONFIG = {
    "language": "English",
    "base_language": "Turkish",  # i18n: Can be changed to "Spanish", "German", etc.
    "txt_path": DATA_DIR / "txt" / "en_example.txt",
    "csv_path": DATA_DIR / "csv" / "en_example.csv",
    "main_csv_path": DATA_DIR / "csv" / "ENGLISH.csv",
    "manual_prompt_path": None,
    "deck_name": "English",
    "note_type": "English_Cloze",
    "field_sentence": "Sentence_English",
    "limit": 21,
    "batch_size": 3,
    "pro_limit": 0,
    "voice": "en-US-JennyNeural",
}


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def extract_json_from_response(raw_response):
    """Parses and extracts a valid JSON object/array from a raw LLM text response."""
    try:
        match = re.search(r"(\{.*\}|\[.*\])", raw_response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("No valid JSON found in the response.")
    except Exception as e:
        logging.warning(f"❌ JSON Parse Error: {e}")
        return None


def wait_for_internet(timeout=60):
    """Pings Google DNS to ensure internet connectivity before starting the pipeline."""
    logging.info("Checking internet connection...")
    start_time = time.time()
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            logging.info("Internet connected. Autonomous pipeline starting!")
            return True
        except OSError:
            if time.time() - start_time > timeout:
                logging.error("Timeout reached. No internet connection. Shutting down.")
                return False
            time.sleep(5)


def extract_words_from_txt(file_path, amount):
    """Extracts a specific batch of words from the master list and returns the remainder."""
    if not os.path.exists(file_path):
        return [], []
    with open(file_path, "r", encoding="utf-8") as f:
        all_words = [s.strip() for s in f.readlines() if s.strip()]
    return all_words[:amount], all_words[amount:]


def update_txt_file(file_path, remaining_words):
    """Safely updates the master list by overwriting it with the remaining words."""
    with open(file_path, "w", encoding="utf-8") as f:
        for word in remaining_words:
            f.write(word + "\n")


def save_to_csv(file_path, flashcards, field_sentence_name):
    """Appends generated flashcards to the daily CSV file (Pipe delimited)."""
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        if not file_exists:
            writer.writerow(
                [field_sentence_name, "Base_Word", "Meaning", "Translation"]
            )

        for card in flashcards:
            sentence = " ".join(card.get("sentence", "").split())
            lemma = " ".join(card.get("base_word", "").split())
            meaning = " ".join(card.get("meaning", "").split())
            translation = " ".join(card.get("translation", "").split())
            writer.writerow([sentence, lemma, meaning, translation])


def safe_api_call(client, model_id, prompt, max_retries=5):
    """
    Executes an API call with an Exponential Backoff strategy.
    If the API hits a Rate Limit (429) or fails, it waits and retries,
    doubling the wait time after each failure to prevent server overload.
    """
    wait_time = 10
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model_id, contents=prompt)
        except Exception as e:
            logging.warning(
                f"⚠️ API Error (Attempt {attempt+1}/{max_retries}): {str(e)}"
            )
            if attempt == max_retries - 1:
                return None
            time.sleep(wait_time)
            wait_time *= 2
    return None


def read_prompt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Load System Prompts Dynamically
DE_PROD_PROMPT = read_prompt_file(PROMPTS_DIR / "de_1_prod.txt")
EN_PROD_PROMPT = read_prompt_file(PROMPTS_DIR / "en_1_prod.txt")
DE_QA_PROMPT = read_prompt_file(PROMPTS_DIR / "de_2_qa.txt")
EN_QA_PROMPT = read_prompt_file(PROMPTS_DIR / "en_2_qa.txt")


# ==============================================================================
# 4. CORE ENGINE (PART 1: GENERATION & QA)
# ==============================================================================
def process_language_batch(client, config, sys_prompt, qa_prompt):
    """Handles the end-to-end automated generation and QA for a specific language."""
    lang = config["language"]
    base_lang = config["base_language"]
    logging.info(f"\n🌍 --- {lang.upper()} PART 1 STARTED ---")

    # Dynamic i18n Prompt Injection
    dynamic_sys_prompt = sys_prompt.replace("[BASE_LANGUAGE]", base_lang)
    dynamic_qa_prompt = qa_prompt.replace("[BASE_LANGUAGE]", base_lang)

    with open(config["csv_path"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(
            [config["field_sentence"], "Base_Word", "Meaning", "Translation"]
        )
        logging.info(f"Daily CSV wiped and ready: {config['csv_path']}")

    total_processed = 0

    while total_processed < config["limit"]:
        batch_words, remaining_words = extract_words_from_txt(
            config["txt_path"], config["batch_size"]
        )
        if not batch_words:
            logging.info(f"✅ No more words left to process for {lang}.")
            break

        batch_no = (total_processed // config["batch_size"]) + 1
        logging.info(f"\n📦 {lang} Batch {batch_no}: {batch_words}")

        # Phase 1: Production (Generation) with dynamic prompt
        gen_response = safe_api_call(
            client,
            MODEL_ID,
            f"{dynamic_sys_prompt}\n\nUSER INPUT: {json.dumps(batch_words)}",
        )
        generated_json = (
            extract_json_from_response(gen_response.text) if gen_response else None
        )

        if not generated_json:
            logging.error(
                f"❌ API/Quota Error during {lang} production. Halting this language."
            )
            break

        # Phase 2: QA (Quality Assurance) with dynamic prompt
        qa_input = json.dumps(generated_json, ensure_ascii=False)
        qa_response = safe_api_call(
            client, MODEL_ID, f"{dynamic_qa_prompt}\n\nRAW CARDS:\n{qa_input}"
        )
        final_json = (
            extract_json_from_response(qa_response.text) if qa_response else None
        )

        if not final_json or not final_json.get("flashcards"):
            logging.error(
                f"❌ API/Quota Error during {lang} QA. Halting this language."
            )
            break

        # Phase 3: Stateful Save
        save_to_csv(
            config["csv_path"], final_json.get("flashcards"), config["field_sentence"]
        )
        update_txt_file(config["txt_path"], remaining_words)

        total_processed += len(batch_words)
        logging.info(
            f"💾 {lang} batch saved to CSV. Total processed: {total_processed}"
        )
        time.sleep(5)

    if config.get("manual_prompt_path") and total_processed > 0:
        desktop = Path.home() / "Desktop"
        manual_file = desktop / f"{lang}_manual_correction.txt"
        generate_manual_correction_file(
            config, config["manual_prompt_path"], manual_file
        )


def generate_manual_correction_file(config, prompt_path, output_path):
    """Compiles today's flawed/generated cards with a prompt for manual Pro-Model review."""
    if not os.path.exists(prompt_path) or not os.path.exists(config["csv_path"]):
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read().strip()

    # Inject language dynamically here as well
    prompt_content = prompt_content.replace("[BASE_LANGUAGE]", config["base_language"])

    data_lines = []
    with open(config["csv_path"], "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                data_lines.append("|".join(row))

    if data_lines:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prompt_content + "\n\n")
            f.write("\n".join(data_lines))
        logging.info(f"📝 Manual correction file generated: {output_path}")


def generate_pro_model_request_file(config):
    """
    Fallback Architecture: Extracts words for manual generation via a more advanced LLM (Pro Model)
    when automated API limits are reached or higher quality is required.
    """
    pro_limit = config.get("pro_limit", 0)
    pro_prompt_path = config.get("pro_prompt_path")

    if pro_limit <= 0 or not pro_prompt_path or not os.path.exists(pro_prompt_path):
        return

    logging.info(
        f"📦 Extracting {pro_limit} words for {config['language']} Pro Model Fallback..."
    )
    pro_words, remaining_words = extract_words_from_txt(config["txt_path"], pro_limit)

    if not pro_words:
        logging.warning(
            f"⚠️ No words left for {config['language']} Pro Model fallback!"
        )
        return

    raw_prompt = read_prompt_file(pro_prompt_path)
    dynamic_prod_prompt = raw_prompt.replace("[BASE_LANGUAGE]", config["base_language"])

    desktop = Path.home() / "Desktop"
    request_file_path = desktop / f"{config['language']}_Pro_Request.txt"

    with open(request_file_path, "w", encoding="utf-8") as f:
        f.write(dynamic_prod_prompt)
        f.write("\n\nUSER INPUT: ")
        f.write(json.dumps(pro_words, ensure_ascii=False))

    update_txt_file(config["txt_path"], remaining_words)
    logging.info(f"📝 Pro Model request file ready on Desktop: {request_file_path}")


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main_pipeline():
    if not wait_for_internet():
        return

    logging.info("🚀 Autonomous Language Factory - PART 1 (Generation) starting...\n")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Step 1: Automated Daily Batch Processing (API)
    logging.info("--- PHASE 1: AUTOMATED API PIPELINE ---")
    process_language_batch(client, GERMAN_CONFIG, DE_PROD_PROMPT, DE_QA_PROMPT)
    process_language_batch(client, ENGLISH_CONFIG, EN_PROD_PROMPT, EN_QA_PROMPT)

    # Step 2: Manual Fallback Generation (Pro Model)
    logging.info("\n--- PHASE 2: HYBRID FALLBACK PIPELINE (PRO MODEL) ---")
    generate_pro_model_request_file(GERMAN_CONFIG)
    generate_pro_model_request_file(ENGLISH_CONFIG)

    logging.info(
        "\n🎉 PART 1 successfully completed! Daily batches and Pro Request files are ready."
    )


if __name__ == "__main__":
    main_pipeline()
