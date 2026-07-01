"""
Completeaza golurile ramase din ingestia Codului Fiscal (cele 259 de bucati care
au esuat din cauza cotei zilnice gratuite Gemini: 1000 cereri embeddings/zi).

Ruleaza acest script DUPA ce se reseteaza cota (de obicei la miezul noptii, ora
Pacific = in jurul orei 9-10 dimineata, ora Romaniei).

Foloseste lista de indici lipsa salvata in lipsa_idx.json si reia procesarea
DOAR pentru acele bucati, cu pauze mai mari si retry cu backoff la eroarea 429.

Ruleaza: python completeaza_goluri.py
"""

import os
import json
import time

from dotenv import load_dotenv
from supabase import create_client
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from ingest_cod_fiscal import (
    extrage_text, imparte_document, EMBED_MODEL, EMBED_DIM,
)

load_dotenv()

PAUZA_INTRE_CERERI = 5  # secunde
BACKOFF_LA_429 = [30, 90, 180]  # secunde, intre incercari succesive


def genereaza_embedding_cu_retry(client: genai.Client, text: str):
    for incercare, asteptare in enumerate([0] + BACKOFF_LA_429):
        if asteptare:
            print(f"    [429] astept {asteptare}s inainte de reincercare {incercare}/{len(BACKOFF_LA_429)}...")
            time.sleep(asteptare)
        try:
            rezultat = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBED_DIM,
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            return rezultat.embeddings[0].values
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                continue
            raise
    return None


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with open("lipsa_idx.json", "r", encoding="utf-8") as f:
        lipsa_idx = set(json.load(f))

    print(f"Bucati de completat: {len(lipsa_idx)}")

    text = extrage_text("codul fiscal.pdf")
    toate = imparte_document(text)

    reusite, esuate = 0, 0
    for i in sorted(lipsa_idx):
        chunk = toate[i]
        embedding = genereaza_embedding_cu_retry(client, chunk["continut"][:18000])

        if embedding is None:
            print(f"  [{i}] ESUAT definitiv (cota inca epuizata?): {chunk['articol'][:60]}")
            esuate += 1
            time.sleep(PAUZA_INTRE_CERERI)
            continue

        sb.table("fiscal_knowledge").insert({
            "continut": chunk["continut"],
            "embedding": embedding,
            "sursa": chunk["sursa"],
            "titlu": chunk["titlu"],
            "articol": chunk["articol"],
            "chunk_index": i,
        }).execute()

        reusite += 1
        print(f"  [{i}] OK: {chunk['articol'][:60]}  ({reusite}/{len(lipsa_idx)})")
        time.sleep(PAUZA_INTRE_CERERI)

    print()
    print(f"Finalizat. Reusite: {reusite}  Esuate: {esuate}")
    if esuate == 0:
        print("Toate golurile au fost completate cu succes!")
    else:
        print("Mai sunt bucati neprocesate - ruleaza scriptul din nou mai tarziu.")


if __name__ == "__main__":
    main()
