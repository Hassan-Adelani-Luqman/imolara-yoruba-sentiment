import unicodedata

from src.data import (clean_tweet, has_tone_marks, has_underdots, nfc, preprocess,
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


def test_preprocess_modes_and_lowercase():
    assert preprocess("Ẹ KÚ ÀBỌ̀", "no_tones", lowercase=True) == "ẹ ku abọ"
    assert preprocess("Ẹ KÚ ÀBỌ̀", "no_diacritics") == "E KU ABO"
