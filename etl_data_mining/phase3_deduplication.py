import os
import random
import logging
from pathlib import Path

# ==============================================================================
# 1. SETUP & PATHS
# ==============================================================================
BASE_DIR = Path(__file__).parent.parent.resolve()
TXT_DIR = BASE_DIR / "data" / "txt"

INPUT_FILE = TXT_DIR / "phase2_normalized_words.txt"
# Output is routed directly to the file that the Main Agent (part_1.py) consumes
OUTPUT_FILE = TXT_DIR / "de_example.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_deduplication():
    logging.info("🧹 Phase 3: Final Deduplication and Shuffling Started...\n")

    if not INPUT_FILE.exists():
        logging.error(f"❌ Error: '{INPUT_FILE.name}' not found!")
        return

    # Read normalized list from Phase 2
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_words = [line.strip() for line in f.readlines() if line.strip()]

    total_input = len(raw_words)

    # 1. Critical Case-Insensitive Deduplication
    unique_words = []
    seen_lower = set()

    for word in raw_words:
        # Convert to lowercase for comparison (e.g., Haus == haus)
        lower_word = word.lower()

        # If we haven't seen this word's lowercase version, add it
        if lower_word not in seen_lower:
            seen_lower.add(lower_word)
            unique_words.append(word)  # Preserve the original capitalized form

    recovered_count = len(unique_words)
    deleted_hidden_duplicates = total_input - recovered_count

    # 2. Algorithmic Shuffling to Reduce Cognitive Interference
    logging.info(f"🔀 Shuffling the list to prevent cognitive interference...")
    random.seed()
    random.shuffle(unique_words)

    # 3. Write The Golden Output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_words))

    # --- FINAL REPORT ---
    logging.info("\n🏆 --- PHASE 3: FINAL (GOLDEN) REPORT ---")
    logging.info(f"📥 Total Input Words      : {total_input}")
    logging.info(f"🗑️ Deleted Hidden Copies : {deleted_hidden_duplicates}")
    logging.info(f"🔥 FINAL GOLDEN LIST      : {recovered_count}")
    logging.info(f"📁 Saved directly to      : {OUTPUT_FILE.name}")
    logging.info(
        "\n🚀 CONGRATULATIONS! Data Engineering pipeline completed. You are ready to run src/part_1.py!"
    )


if __name__ == "__main__":
    run_deduplication()
