"""
Kokoro-82M Symbol Mapping for StyleTTS2
=========================================
Auto-generated from Kokoro-82M config.json.
Replaces StyleTTS2's default symbol list in text_utils.py and meldataset.py.

CRITICAL: Kokoro and StyleTTS2 use different index assignments for the same
178-token vocabulary. This file provides the exact mapping that matches
Kokoro-82M's pre-trained embeddings.

Usage in StyleTTS2:
  from kokoro_symbols import symbols, dicts
"""

# fmt: off
# Kokoro-82M vocabulary: 178 tokens (index 0 = pad, gaps filled with PUA chars)
symbols = [
    '$',  #   0: PAD
    ';',  #   1: ;
    ':',  #   2: :
    ',',  #   3: ,
    '.',  #   4: .
    '!',  #   5: !
    '?',  #   6: ?
    '\ue000',  #   7: (unused placeholder)
    '\ue001',  #   8: (unused placeholder)
    '—',  #   9: U+2014 (—)
    '…',  #  10: U+2026 (…)
    '\"',  #  11: "
    '(',  #  12: (
    ')',  #  13: )
    '“',  #  14: U+201C (“)
    '”',  #  15: U+201D (”)
    ' ',  #  16:  
    '̃',  #  17: U+0303 (̃)
    'ʣ',  #  18: U+02A3 (ʣ)
    'ʥ',  #  19: U+02A5 (ʥ)
    'ʦ',  #  20: U+02A6 (ʦ)
    'ʨ',  #  21: U+02A8 (ʨ)
    'ᵝ',  #  22: U+1D5D (ᵝ)
    'ꭧ',  #  23: U+AB67 (ꭧ)
    'A',  #  24: A
    'I',  #  25: I
    '\ue002',  #  26: (unused placeholder)
    '\ue003',  #  27: (unused placeholder)
    '\ue004',  #  28: (unused placeholder)
    '\ue005',  #  29: (unused placeholder)
    '\ue006',  #  30: (unused placeholder)
    'O',  #  31: O
    '\ue007',  #  32: (unused placeholder)
    'Q',  #  33: Q
    '\ue008',  #  34: (unused placeholder)
    'S',  #  35: S
    'T',  #  36: T
    '\ue009',  #  37: (unused placeholder)
    '\ue00a',  #  38: (unused placeholder)
    'W',  #  39: W
    '\ue00b',  #  40: (unused placeholder)
    'Y',  #  41: Y
    'ᵊ',  #  42: U+1D4A (ᵊ)
    'a',  #  43: a
    'b',  #  44: b
    'c',  #  45: c
    'd',  #  46: d
    'e',  #  47: e
    'f',  #  48: f
    '\ue00c',  #  49: (unused placeholder)
    'h',  #  50: h
    'i',  #  51: i
    'j',  #  52: j
    'k',  #  53: k
    'l',  #  54: l
    'm',  #  55: m
    'n',  #  56: n
    'o',  #  57: o
    'p',  #  58: p
    'q',  #  59: q
    'r',  #  60: r
    's',  #  61: s
    't',  #  62: t
    'u',  #  63: u
    'v',  #  64: v
    'w',  #  65: w
    'x',  #  66: x
    'y',  #  67: y
    'z',  #  68: z
    'ɑ',  #  69: U+0251 (ɑ)
    'ɐ',  #  70: U+0250 (ɐ)
    'ɒ',  #  71: U+0252 (ɒ)
    'æ',  #  72: U+00E6 (æ)
    '\ue00d',  #  73: (unused placeholder)
    '\ue00e',  #  74: (unused placeholder)
    'β',  #  75: U+03B2 (β)
    'ɔ',  #  76: U+0254 (ɔ)
    'ɕ',  #  77: U+0255 (ɕ)
    'ç',  #  78: U+00E7 (ç)
    '\ue00f',  #  79: (unused placeholder)
    'ɖ',  #  80: U+0256 (ɖ)
    'ð',  #  81: U+00F0 (ð)
    'ʤ',  #  82: U+02A4 (ʤ)
    'ə',  #  83: U+0259 (ə)
    '\ue010',  #  84: (unused placeholder)
    'ɚ',  #  85: U+025A (ɚ)
    'ɛ',  #  86: U+025B (ɛ)
    'ɜ',  #  87: U+025C (ɜ)
    '\ue011',  #  88: (unused placeholder)
    '\ue012',  #  89: (unused placeholder)
    'ɟ',  #  90: U+025F (ɟ)
    '\ue013',  #  91: (unused placeholder)
    'ɡ',  #  92: U+0261 (ɡ)
    '\ue014',  #  93: (unused placeholder)
    '\ue015',  #  94: (unused placeholder)
    '\ue016',  #  95: (unused placeholder)
    '\ue017',  #  96: (unused placeholder)
    '\ue018',  #  97: (unused placeholder)
    '\ue019',  #  98: (unused placeholder)
    'ɥ',  #  99: U+0265 (ɥ)
    '\ue01a',  # 100: (unused placeholder)
    'ɨ',  # 101: U+0268 (ɨ)
    'ɪ',  # 102: U+026A (ɪ)
    'ʝ',  # 103: U+029D (ʝ)
    '\ue01b',  # 104: (unused placeholder)
    '\ue01c',  # 105: (unused placeholder)
    '\ue01d',  # 106: (unused placeholder)
    '\ue01e',  # 107: (unused placeholder)
    '\ue01f',  # 108: (unused placeholder)
    '\ue020',  # 109: (unused placeholder)
    'ɯ',  # 110: U+026F (ɯ)
    'ɰ',  # 111: U+0270 (ɰ)
    'ŋ',  # 112: U+014B (ŋ)
    'ɳ',  # 113: U+0273 (ɳ)
    'ɲ',  # 114: U+0272 (ɲ)
    'ɴ',  # 115: U+0274 (ɴ)
    'ø',  # 116: U+00F8 (ø)
    '\ue021',  # 117: (unused placeholder)
    'ɸ',  # 118: U+0278 (ɸ)
    'θ',  # 119: U+03B8 (θ)
    'œ',  # 120: U+0153 (œ)
    '\ue022',  # 121: (unused placeholder)
    '\ue023',  # 122: (unused placeholder)
    'ɹ',  # 123: U+0279 (ɹ)
    '\ue024',  # 124: (unused placeholder)
    'ɾ',  # 125: U+027E (ɾ)
    'ɻ',  # 126: U+027B (ɻ)
    '\ue025',  # 127: (unused placeholder)
    'ʁ',  # 128: U+0281 (ʁ)
    'ɽ',  # 129: U+027D (ɽ)
    'ʂ',  # 130: U+0282 (ʂ)
    'ʃ',  # 131: U+0283 (ʃ)
    'ʈ',  # 132: U+0288 (ʈ)
    'ʧ',  # 133: U+02A7 (ʧ)
    '\ue026',  # 134: (unused placeholder)
    'ʊ',  # 135: U+028A (ʊ)
    'ʋ',  # 136: U+028B (ʋ)
    '\ue027',  # 137: (unused placeholder)
    'ʌ',  # 138: U+028C (ʌ)
    'ɣ',  # 139: U+0263 (ɣ)
    'ɤ',  # 140: U+0264 (ɤ)
    '\ue028',  # 141: (unused placeholder)
    'χ',  # 142: U+03C7 (χ)
    'ʎ',  # 143: U+028E (ʎ)
    '\ue029',  # 144: (unused placeholder)
    '\ue02a',  # 145: (unused placeholder)
    '\ue02b',  # 146: (unused placeholder)
    'ʒ',  # 147: U+0292 (ʒ)
    'ʔ',  # 148: U+0294 (ʔ)
    '\ue02c',  # 149: (unused placeholder)
    '\ue02d',  # 150: (unused placeholder)
    '\ue02e',  # 151: (unused placeholder)
    '\ue02f',  # 152: (unused placeholder)
    '\ue030',  # 153: (unused placeholder)
    '\ue031',  # 154: (unused placeholder)
    '\ue032',  # 155: (unused placeholder)
    'ˈ',  # 156: U+02C8 (ˈ)
    'ˌ',  # 157: U+02CC (ˌ)
    'ː',  # 158: U+02D0 (ː)
    '\ue033',  # 159: (unused placeholder)
    '\ue034',  # 160: (unused placeholder)
    '\ue035',  # 161: (unused placeholder)
    'ʰ',  # 162: U+02B0 (ʰ)
    '\ue036',  # 163: (unused placeholder)
    'ʲ',  # 164: U+02B2 (ʲ)
    '\ue037',  # 165: (unused placeholder)
    '\ue038',  # 166: (unused placeholder)
    '\ue039',  # 167: (unused placeholder)
    '\ue03a',  # 168: (unused placeholder)
    '↓',  # 169: U+2193 (↓)
    '\ue03b',  # 170: (unused placeholder)
    '→',  # 171: U+2192 (→)
    '↗',  # 172: U+2197 (↗)
    '↘',  # 173: U+2198 (↘)
    '\ue03c',  # 174: (unused placeholder)
    '\ue03d',  # 175: (unused placeholder)
    '\ue03e',  # 176: (unused placeholder)
    'ᵻ',  # 177: U+1D7B (ᵻ)
]
# fmt: on

# Build symbol-to-ID lookup dict (same interface as StyleTTS2's TextCleaner)
dicts = {sym: i for i, sym in enumerate(symbols)}


class TextCleaner:
    """Drop-in replacement for StyleTTS2's TextCleaner using Kokoro vocab."""
    def __init__(self, dummy=0):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        # Map each character to its index, skip unknown chars
        indexes = []
        for char in text:
            if char in self.word_index_dictionary:
                indexes.append(self.word_index_dictionary[char])
        return indexes


assert len(symbols) == 178, f"Expected 178 symbols, got {len(symbols)}"

