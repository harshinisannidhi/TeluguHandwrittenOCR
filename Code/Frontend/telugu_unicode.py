
import re

TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
VIRAMA = "\u0C4D"

VOWELS = {  # achulu
    "a": "అ", "aa": "ఆ", "i": "ఇ", "ii": "ఈ", "u": "ఉ", "uu": "ఊ", "ru": "ఋ", "ruu": "ౠ",
    "e": "ఎ", "ee": "ఏ", "ai": "ఐ", "o": "ఒ", "oo": "ఓ", "au": "ఔ", "am": "అం", "ah": "అః",
}
VOWEL_SIGNS = {  # used for guninthamulu (consonant + vowel sign); "a" is the inherent vowel
    "a": "", "aa": "ా", "i": "ి", "ii": "ీ", "u": "ు", "uu": "ూ", "ru": "ృ", "ruu": "ౄ",
    "e": "ె", "ee": "ే", "ai": "ై", "o": "ొ", "oo": "ో", "au": "ౌ", "am": "ం", "ah": "ః",
}
CONSONANT_STEMS = {  # hallulu (stem without the inherent 'a')
    "k": "క", "kh": "ఖ", "g": "గ", "gh": "ఘ", "ng": "ఙ", "c": "చ", "ch": "చ", "chh": "ఛ",
    "j": "జ", "jh": "ఝ", "ny": "ఞ", "tt": "ట", "tth": "ఠ", "dd": "డ", "ddh": "ఢ", "nn": "ణ",
    "t": "త", "th": "థ", "d": "ద", "dh": "ధ", "n": "న", "p": "ప", "ph": "ఫ", "b": "బ",
    "bh": "భ", "m": "మ", "y": "య", "r": "ర", "rr": "ఱ", "l": "ల", "v": "వ", "w": "వ",
    "sh": "శ", "ss": "ష", "s": "స", "h": "హ", "ll": "ళ", "ksh": "క్ష",
}
MARKERS = ("gunintham", "gunintha", "guninthamulu", "othulu", "othu", "vattulu", "vattu",
           "hallulu", "achulu", "halu", "achu")

def resolve_unicode(name, group=""):
    # returns (unicode_string, resolved_flag)
    if TELUGU_RE.search(name):
        return name.strip(), True
    low = re.sub(r"[^a-z]", "", name.lower())
    is_othu = ("othu" in low) or ("vattu" in low) or group.lower().startswith("othu")
    for m in sorted(MARKERS, key=len, reverse=True):
        low = low.replace(m, "")
    if not low:
        return name, False
    if group.lower().startswith("achu") and low in VOWELS:
        return VOWELS[low], True
    for stem in sorted(CONSONANT_STEMS, key=len, reverse=True):
        if low.startswith(stem):
            rest = low[len(stem):] or "a"
            if rest in VOWEL_SIGNS:
                base = CONSONANT_STEMS[stem]
                if base == "క్ష":
                    cons = base
                else:
                    cons = base
                if is_othu:
                    return VIRAMA + cons + VOWEL_SIGNS[rest], True
                return cons + VOWEL_SIGNS[rest], True
    if low in VOWELS:
        return VOWELS[low], True
    return name, False
