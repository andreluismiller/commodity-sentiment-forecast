import pandas as pd
from src.common import normalize_key,sha256_text
from src.nlp.event_classifier import classify
from src.common import load_yaml

def test_normalize():assert normalize_key("Aço Elétrico!")=="aco eletrico"
def test_hash_stable():assert sha256_text(" Cobre ")==sha256_text("cobre")
def test_event():
    events=load_yaml("config/keywords.yaml")["events"];result=classify("Strike causes supply disruption at copper mine",events);assert result["event_price_impact"]==1
