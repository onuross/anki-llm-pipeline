import re
import csv
import logging
from pathlib import Path

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dynamically resolve paths based on the new folder structure
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"


# ==============================================================================
# 2. CORE LOGIC (Decoupled for Testability)
# ==============================================================================
def process_cloze_sentence(sentence: str, meaning: str) -> str:
    """
    Parses HTML bold tags in a sentence and converts them into Anki Cloze format.
    Applies a merging mechanism for adjacent <b> tags before wrapping.
    """
    # "Japanese Glue": Merge adjacent <b> tags (e.g., <b>aus</b><b>ge</b> -> <b>ausge)
    sentence = re.sub(r"</b>\s*<b>", "", sentence)

    matches = re.findall(r"<b>(.*?)</b>", sentence)
    if not matches:
        return sentence

    if len(matches) == 1:
        # Single match: Standard cloze wrap
        return re.sub(r"<b>(.*?)</b>", rf"{{{{c1::\1::{meaning}}}}}", sentence)

    # Multiple matches: Find the longest word to attach the main meaning hint
    longest_match = max(matches, key=len)
    processed_sentence = sentence

    for match in matches:
        if match == longest_match:
            # Attach the hint to the main verb / longest part
            processed_sentence = processed_sentence.replace(
                f"<b>{match}</b>", f"{{{{c1::{match}::{meaning}}}}}", 1
            )
        else:
            # Prefix/suffix parts get the cloze tag without the hint
            processed_sentence = processed_sentence.replace(
                f"<b>{match}</b>", f"{{{{c1::{match}}}}}", 1
            )

    return processed_sentence


# ==============================================================================
# 3. FILE I/O HANDLER
# ==============================================================================
def convert_to_cloze_cards(input_path: Path, output_path: Path):
    """
    Reads a pipe-separated CSV, converts targeted sentences to Anki cloze format,
    and saves the output securely to a new file.
    """
    if not input_path.exists():
        logging.error(f"❌ Input file not found: {input_path}")
        return

    successful_conversions = 0

    # Using the csv module is much safer than manual string splitting
    with open(input_path, "r", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.reader(infile, delimiter="|")
        writer = csv.writer(outfile, delimiter="|")

        for row in reader:
            # Clean up whitespace from all columns
            row = [col.strip() for col in row]

            if len(row) == 4:
                lemma, meaning, sentence, translation = row

                new_sentence = process_cloze_sentence(sentence, meaning)
                writer.writerow([lemma, meaning, new_sentence, translation])
                successful_conversions += 1
            else:
                # Write malformed rows as they are to prevent data loss
                writer.writerow(row)

    logging.info("🎉 Conversion Completed!")
    logging.info(
        f"✅ {successful_conversions} cards successfully converted to Cloze format."
    )
    logging.info(f"📁 Your file is ready for Anki injection: {output_path.name}")


# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Define input and output files (adjust the filenames as per your data folder)
    INPUT_CSV = DATA_DIR / "csv" / "raw_cards.csv"
    OUTPUT_CSV = DATA_DIR / "csv" / "cloze_ready_cards.csv"

    convert_to_cloze_cards(INPUT_CSV, OUTPUT_CSV)
