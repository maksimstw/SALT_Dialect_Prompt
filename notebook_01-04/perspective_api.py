from googleapiclient import discovery
from tqdm import tqdm
import json
import pandas as pd
import time


API_KEY = 'AIzaSyB2ptgd5lBDnpjjXc9vP2I9MAi2c75tXRQ'

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


def make_request(comment):
    analyze_request = {
        'comment': {'text': comment},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': 'en'
    }
    response = client.comments().analyze(body=analyze_request).execute()
    return response


df = pd.read_csv("runs/03 EMNLP SAE-AAVE Pairs/sae_gen_txt.csv")
with open('runs/03 EMNLP SAE-AAVE Pairs/Toxicity/sae_gen_txt_toxicity.json', 'a', encoding="utf-8") as f:
    for txt in tqdm(df.txt):
        json.dump(make_request(txt), f, indent=2)
        f.write(',')
        time.sleep(1)
