import pandas as pd
from deep_translator import GoogleTranslator
from difflib import SequenceMatcher
import time

def fast_translate_and_evaluate(input_file, output_file):
    df = pd.read_csv(input_file)

    tr_to_en = GoogleTranslator(source='tr', target='en')
    en_to_tr = GoogleTranslator(source='en', target='tr')

    translated_en_comments = []
    back_translated_tr_comments = []
    similarity_scores = []

    for idx, row in df.iterrows():
        try:
            original_tr = str(row['Comment'])

            # Translate once and reuse
            translated_en = tr_to_en.translate(original_tr)
            back_translated_tr = en_to_tr.translate(translated_en)
            similarity = SequenceMatcher(None, original_tr, back_translated_tr).ratio()

            translated_en_comments.append(translated_en)
            back_translated_tr_comments.append(back_translated_tr)
            similarity_scores.append(similarity)

            print(f"Row {idx+1}/{len(df)} | Sim: {similarity:.2f}")
            # time.sleep(0.1)  # optional: adjust if rate limited

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            translated_en_comments.append("")
            back_translated_tr_comments.append("")
            similarity_scores.append(0.0)

    df['Comment_en'] = translated_en_comments
    df['BackTranslated_TR'] = back_translated_tr_comments
    df['SimilarityScore'] = similarity_scores

    df.to_csv(output_file, index=False)
    print(f"Saved translated file to: {output_file}")

# Usage
fast_translate_and_evaluate(
    input_file='../datasets/exterim/electronicComments.csv',
    output_file='../datasets/interim/electronicComments_evaluation_fast.csv'
)
