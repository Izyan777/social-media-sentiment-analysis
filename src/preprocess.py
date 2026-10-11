"""Cleaning + Roman Urdu normalisation.

Roman Urdu has no standard spelling ("nahi", "nhi", "nai", "nahin" are all the same word),
so the biggest win in preprocessing is collapsing common variants onto one form.
"""
import json
import re

import pandas as pd

from src.config import CLEAN_CSV, PROCESSED_DIR, ROMAN_URDU_CSV, ensure_dirs

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # "wahhhh" -> "wahh"
WS_RE = re.compile(r"\s+")

# Common Roman Urdu spelling variants -> one canonical form.
# Hand-built from frequent informal spellings; extend freely.
VARIANTS = {
    # negation / basics
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "nahee": "nahi", "nae": "nahi", "nhn": "nahi",
    "hy": "hai", "hay": "hai", "hain": "hai", "hn": "han", "haan": "han", "jee": "ji",
    "bht": "bohat", "bohot": "bohat", "buhat": "bohat", "boht": "bohat", "bhot": "bohat",
    "buhut": "bohat", "bohut": "bohat", "bahut": "bohat", "bhut": "bohat",
    "acha": "acha", "achha": "acha", "accha": "acha", "achaa": "acha", "achi": "acha", "achay": "acha",
    "bura": "bura", "buri": "bura", "bure": "bura",
    "kuch": "kuch", "kch": "kuch", "kuchh": "kuch",
    "kya": "kya", "kia": "kya", "kiya": "kya",
    "kyun": "kyun", "kyu": "kyun", "kiun": "kyun", "q": "kyun",
    "mujhe": "mujhe", "mujy": "mujhe", "mjhe": "mujhe", "mujhay": "mujhe", "mjhy": "mujhe",
    "tum": "tum", "tm": "tum", "aap": "aap", "ap": "aap",
    "k": "ke", "ka": "ka", "ki": "ki", "ko": "ko", "se": "se", "sy": "se", "say": "se",
    "tou": "to", "tho": "to", "tow": "to",  # NOTE: "tu" (= "you") is deliberately NOT mapped
    "aur": "aur", "or": "aur", "ur": "aur",
    "bhi": "bhi", "b": "bhi", "bi": "bhi",
    "lekin": "lekin", "lakin": "lekin", "magar": "magar", "par": "par", "per": "par", "pr": "par",
    "kyunke": "kyunke", "kyunki": "kyunke", "kuki": "kyunke",
    "thanks": "shukriya", "thank": "shukriya", "shukria": "shukriya", "shukriyah": "shukriya",
    "shukriya": "shukriya", "shukar": "shukar", "mashallah": "mashaallah", "masha": "mashaallah",
    "mashaallah": "mashaallah", "inshallah": "inshaallah", "inshaallah": "inshaallah",
    # praise
    "zabardast": "zabardast", "zabrdast": "zabardast", "zbrdast": "zabardast", "zaberdast": "zabardast",
    "behtareen": "behtareen", "behtrin": "behtareen", "behtreen": "behtareen",
    "umda": "umda", "umdah": "umda", "kamaal": "kamal", "kamal": "kamal",
    "pasand": "pasand", "psnd": "pasand",
    # complaints
    "kharab": "kharab", "kharaab": "kharab", "kharb": "kharab", "kherab": "kharab",
    "bakwas": "bakwas", "bakwaas": "bakwas", "bkwas": "bakwas",
    "ghatiya": "ghatiya", "ghatia": "ghatiya", "ghatya": "ghatiya",
    "mehnga": "mehnga", "mehngi": "mehnga", "mehnge": "mehnga", "mahanga": "mehnga",
    "mahngi": "mehnga", "mehngai": "mehngai", "mehngayi": "mehngai", "mahngai": "mehngai",
    "nakli": "nakli", "naqli": "nakli", "fake": "nakli", "duplicate": "nakli",
    "dhoka": "dhoka", "dhokha": "dhoka", "fraud": "dhoka", "scam": "dhoka",
    "wapas": "wapas", "wapis": "wapas", "wapsi": "wapas", "waapas": "wapas",
    "kamzor": "kamzor", "kamzoor": "kamzor",
    # commerce words
    "delivery": "delivery", "dilevery": "delivery", "delivry": "delivery", "deliveri": "delivery",
    "delivered": "delivery", "deliver": "delivery",
    "parcel": "parcel", "parsal": "parcel", "parcal": "parcel", "parsel": "parcel",
    "packing": "packing", "packaging": "packing", "pakistan": "pakistan", "pak": "pak",
    "quality": "quality", "qualty": "quality", "quailty": "quality", "qaulity": "quality",
    "price": "price", "prise": "price", "qeemat": "qeemat", "keemat": "qeemat", "qimat": "qeemat",
    "service": "service", "sarvis": "service", "servis": "service",
    "customer": "customer", "custmer": "customer", "costumer": "customer",
    "refund": "refund", "rifund": "refund",
    "order": "order", "ordar": "order", "oder": "order",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip URLs/@mentions/punctuation, collapse repeats, unify spellings."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    t = REPEAT_RE.sub(r"\1\1", t)
    t = NON_TEXT_RE.sub(" ", t)
    tokens = [VARIANTS.get(tok, tok) for tok in t.split()]
    return WS_RE.sub(" ", " ".join(tokens)).strip()


def load_and_clean(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the raw Roman Urdu CSV and return (clean_df, cleaning_log)."""
    ensure_dirs()
    if not ROMAN_URDU_CSV.exists():
        raise FileNotFoundError(f"{ROMAN_URDU_CSV} not found. Run:  python -m src.download_data")

    # File has no header: text, label, (empty trailing column)
    raw = pd.read_csv(ROMAN_URDU_CSV, header=None, names=["text", "label", "_extra"], encoding="utf-8")
    log = {"raw_rows": int(len(raw))}

    df = raw[["text", "label"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.title().replace({"Neative": "Negative"})

    keep = df["label"].isin(["Negative", "Neutral", "Positive"]) & df["text"].notna()
    log["dropped_invalid_label_or_missing_text"] = int((~keep).sum())
    df = df[keep].copy()

    df["clean_text"] = df["text"].astype(str).map(normalize_text)

    too_short = df["clean_text"].str.count(r"[a-z]") < 2
    log["dropped_empty_or_junk_text"] = int(too_short.sum())
    df = df[~too_short].copy()

    # Same text with different labels = annotation noise; drop every copy.
    n_labels = df.groupby("clean_text")["label"].transform("nunique")
    conflicting = n_labels > 1
    log["dropped_conflicting_label_rows"] = int(conflicting.sum())
    df = df[~conflicting].copy()

    before = len(df)
    df = df.drop_duplicates(subset="clean_text", keep="first").copy()
    log["dropped_duplicate_rows"] = int(before - len(df))

    df["n_words"] = df["clean_text"].str.split().str.len()
    df = df.reset_index(drop=True)
    log["final_rows"] = int(len(df))
    log["label_counts"] = {k: int(v) for k, v in df["label"].value_counts().items()}

    if save:
        df.to_csv(CLEAN_CSV, index=False)
        (PROCESSED_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    return df, log


if __name__ == "__main__":
    out, lg = load_and_clean()
    print(json.dumps(lg, indent=2))
    print(out.head(8).to_string())
