import unicodedata

from src.data import (clean_tweet, has_tone_marks, has_underdots, nfc, normalize_semeval, preprocess,
                      strip_all_diacritics, strip_tones)


def test_nfc_unifies_composed_and_decomposed_forms():
    decomposed = unicodedata.normalize("NFD", "ẹ̀kọ́")
    composed = unicodedata.normalize("NFC", "ẹ̀kọ́")
    assert decomposed != composed
    assert nfc(decomposed) == composed


def test_strip_tones_keeps_underdots():
    assert strip_tones("Yorùbá ọ̀rọ̀ ṣé") == "Yoruba ọrọ ṣe"


def test_strip_all_diacritics_removes_underdots_too():
    assert strip_all_diacritics("Yorùbá ọ̀rọ̀ ṣé") == "Yoruba oro se"


def test_diacritic_detection():
    assert has_tone_marks("àbọ̀") and has_underdots("àbọ̀")
    assert not has_tone_marks("ọrọ") and has_underdots("ọrọ")
    assert not has_tone_marks("oro") and not has_underdots("oro")


def test_clean_tweet_masks_users_urls_and_repeats():
    text = "@Ayo   wo https://t.co/x  ẹ kú iṣẹ́ ooooo!!!!!! 😂😂😂😂😂"
    assert clean_tweet(text) == "@user wo http ẹ kú iṣẹ́ ooo!!! 😂😂😂"


def test_normalize_semeval_matches_organiser_test_format():
    # real train tweet and its near-duplicate in the released test split
    raw = "@user kòlè yé e yín. Àmọ́ mo mọ̀ wípé á á dára. E lè dá 'promo' dúró fún 'gbà díẹ̀. Ẹ dákun, ẹ mọ́ bínú, ó kú díẹ̀ :)"
    test = "kòlè yé e yín àmọ́ mo mọ̀ wípé á á dára e lè dá promo dúró fún gbà díẹ̀ ẹ dákun ẹ mọ́ bínú ó kú díẹ̀"
    assert normalize_semeval(nfc(raw)) == nfc(test)


def test_normalize_semeval_drops_hashtags_urls_rt_digits_emoji():
    raw = "RT @user: Ẹ KÚ ỌDÚN 2024 🎄❤️ #TweetInYoruba http://t.co/x"
    assert normalize_semeval(nfc(raw)) == nfc("ẹ kú ọdún")


def test_preprocess_styles_and_diacritic_modes():
    assert preprocess("Ẹ KÚ ÀBỌ̀!", "no_tones") == "ẹ ku abọ"
    assert preprocess("Ẹ KÚ ÀBỌ̀!", "no_diacritics", style="raw") == "E KU ABO!"
