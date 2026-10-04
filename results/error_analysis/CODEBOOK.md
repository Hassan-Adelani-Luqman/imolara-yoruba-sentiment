# Error-analysis codebook

Used to tag misclassified test tweets (`errors_tagged.csv`) and to verify those tags (`verification_sheet.csv`).
A tweet can have **several** categories, separated by `;`. Tag what explains the *error*, not everything in the tweet.

| Category | Use when… | Example (from the data) |
|---|---|---|
| **proverb / idiom** | The sentiment is carried by a proverb (òwe), idiom, riddle or poetic figure whose literal words point elsewhere | *àkàrà tú sépo*: "the bean cake dissolved in the oil", i.e. things fell apart (negative) |
| **code-switching** | English (or Pidgin) words matter for the meaning or confuse the model | *how much water should we drink in a day ẹ báwa dási* |
| **missing diacritics / ambiguity** | The tweet is written without tone marks / under-dots (or with non-standard ones), and that plausibly causes the confusion | *ogede dudu o ya nbushan omo buruku…* |
| **sarcasm or irony** | The literal meaning is the opposite of the intended one | *Super Eagles … or super chicken* |
| **slang / new word** | Slang, typos, stylised Unicode letters or rare spellings | *τ̲̅ơ̴̴̴͡ ba jumilo…* |
| **religious or greeting formula** | Religious text, prayers or greetings whose positive *words* do not decide the label (or that the model over-reads as positive) | Genesis 1:1 *ní àtètèkọ́ṣe Ọlọ́run dá ọ̀run àti ayé* (gold neutral) |
| **news / factual (neutral vs other)** | Reports, history, weather, recipes, cultural explanations: neutral style about a positive or negative topic | weather report; Boko Haram rescue news |
| **neutral–negative boundary** | It is genuinely debatable whether the tweet is neutral or negative (or neutral vs positive) | educational quiz about wicked charms |
| **likely label noise** | You believe the gold label is wrong (state the label you would give in `notes`) | *oṣó yí mi ká* "wizards surround me", gold positive |
| **too short / no context** | Too few words, or a fragment, to decide the sentiment | *lo gbe konga* |
| **other** | Anything else (say what in `notes`) | word-of-the-day lessons; oríkì praise poetry |

**Confidence** (for the original tags): `high` = the meaning is clear; `medium` = the gist is clear, details uncertain; `low` = the meaning itself is uncertain.

**How to verify** (`verification_sheet.csv`):
1. Read `normalised_text` (and `raw_text` if present) and the `gold` label.
2. Fill `human_category` with the categories *you* would choose (use this codebook; `;`-separated). Do this before looking at `claude_category`, if possible.
3. Put `Y` or `N` in `gold_label_ok` (do you agree with the annotators' gold label?) and add a gloss or correction in `human_notes`.
4. Run `python -m src.tag_agreement` to compute agreement between the model-assisted tags and yours.
