"""
Actualizeaza baza de legi pentru partea de angajare/recrutare.

- salveaza local fragmentele care se sterg (backup_fragmente_*.json)
- sterge HG 500/2011 (depasita) si vechile fragmente "HG 1425" (erau textul Legii 319)
- incarca actele noi; poate fi repornit, sare peste fragmentele deja salvate

Rulare:
    python ingest_recrutare.py --proba   (nu modifica nimic)
    python ingest_recrutare.py
"""

import json
import os
import re
import sys
import time
from datetime import date

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from supabase import create_client

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
MAX_CHUNK = 6000
SUPRAPUNERE = 500
PAUZA = 1
REINCERCARI = [5, 20, 60, 120]

SURSE_DE_STERS = [
    "HG 1425/2006 - Norme metodologice SSM",
    "HG 500/2011 - Registrul General de Evidență a Salariaților (REVISAL)",
]

DOCUMENTE = [
    {
        "pdf": "HG 295 din 2025.pdf",
        "sursa": "HG 295/2025 - Registrul general de evidență a salariaților (REGES-ONLINE)",
    },
    {
        "pdf": "HG 355 din  2007 privind supravegherea sănătății lucrătorilor.pdf",
        "sursa": "HG 355/2007 - Supravegherea sănătății lucrătorilor (control medical)",
    },
    {
        "pdf": "LEGE nr. 190 din 18 iulie 2018.pdf",
        "sursa": "Legea 190/2018 - Aplicarea GDPR în România",
    },
    {
        "pdf": "ORDONANȚĂ nr. 25 din 2014.pdf",
        "sursa": "OG 25/2014 - Încadrarea în muncă și detașarea străinilor",
    },
    {
        "pdf": "HOTĂRÂRE nr. 1425 din 11 octombrie 2006.pdf",
        "sursa": "HG 1425/2006 - Norme metodologice SSM (text consolidat doar până la 27.12.2011; modificat ulterior prin HG 259/2022)",
    },
    {
        "pdf": "HOTĂRÂRE nr. 259 din 23 februarie 2022.pdf",
        "sursa": "HG 259/2022 - Modificări la Normele metodologice SSM (HG 1425/2006)",
    },
]

# Doar la inceput de rand: notele din textele consolidate ("..., Articolul 260, Capitolul IV")
# nu trebuie sa rupa un articol in doua.
ARTICOL_RE = re.compile(r"^[ \t]*(?:ART\.?[ \t]*|Articolul[ \t]+)(\d+(?:\^\d+)?)[ \t]*[-–.]?[ \t]*([^\n]*)$", re.MULTILINE)
TITLU_RE = re.compile(r"^[ \t]*(?:TITLUL|CAPITOLUL|SECTIUNEA|SECȚIUNEA)[ \t]+\S+[ \t]*[-–]?[ \t]*([^\n]*)$", re.IGNORECASE | re.MULTILINE)


def extrage_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def imparte_articole(text: str, sursa: str) -> list[dict]:
    pozitii = list(ARTICOL_RE.finditer(text))
    if not pozitii:
        raise ValueError(f"Niciun articol gasit in: {sursa}")

    titluri = [(m.start(), m.group(1).strip()[:100]) for m in TITLU_RE.finditer(text)]
    chunks = []
    for idx, m in enumerate(pozitii):
        end = pozitii[idx + 1].start() if idx + 1 < len(pozitii) else len(text)
        bloc = text[m.start():end].strip()
        if len(bloc) < 20:
            continue
        titlu_art = m.group(2).strip()[:120]
        titlu_sectiune = next((t for p, t in reversed(titluri) if p < m.start()), "")
        chunks.append({
            "continut": bloc,
            "articol": f"Art. {m.group(1)}" + (f" - {titlu_art}" if titlu_art else ""),
            "titlu": titlu_sectiune,
            "sursa": sursa,
        })
    return chunks


def sub_imparte(chunk: dict) -> list[dict]:
    continut = chunk["continut"]
    if len(continut) <= MAX_CHUNK:
        return [chunk]
    fragmente = []
    start = 0
    parte = 1
    while start < len(continut):
        end = min(start + MAX_CHUNK, len(continut))
        if end < len(continut):
            taietura = continut.rfind("\n", start + 1000, end)
            if taietura != -1:
                end = taietura
        bucata = continut[start:end].strip()
        if bucata:
            fragmente.append({**chunk, "continut": bucata, "articol": f"{chunk['articol']} (partea {parte})"})
            parte += 1
        if end >= len(continut):
            break
        start = end - SUPRAPUNERE
    return fragmente


def pregateste(doc: dict) -> list[dict]:
    chunks = []
    for c in imparte_articole(extrage_text(doc["pdf"]), doc["sursa"]):
        chunks.extend(sub_imparte(c))
    return chunks


def toate_randurile(sb, sursa: str, coloane: str) -> list[dict]:
    randuri, start = [], 0
    while True:
        pagina = sb.table("fiscal_knowledge").select(coloane).eq("sursa", sursa).range(start, start + 499).execute().data
        randuri += pagina
        start += 500
        if len(pagina) < 500:
            return randuri


def backup_si_sterge(sb) -> None:
    fisier = f"backup_fragmente_{date.today().isoformat()}.json"
    salvate = json.load(open(fisier, encoding="utf-8")) if os.path.exists(fisier) else []
    for sursa in SURSE_DE_STERS:
        randuri = toate_randurile(sb, sursa, "*")
        if not randuri:
            print(f"Nimic de sters: {sursa}")
            continue
        salvate += randuri
        with open(fisier, "w", encoding="utf-8") as f:
            json.dump(salvate, f, ensure_ascii=False)
        sb.table("fiscal_knowledge").delete().eq("sursa", sursa).execute()
        print(f"Sters {len(randuri)} fragmente (backup in {fisier}): {sursa}")


def embedding_cu_reincercari(client, text: str) -> list[float]:
    for pauza in [0] + REINCERCARI:
        if pauza:
            print(f"  reincerc in {pauza}s...", flush=True)
            time.sleep(pauza)
        try:
            r = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text[:18000],
                config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"),
            )
            return r.embeddings[0].values
        except Exception as e:
            ultima = e
            print(f"  eroare embedding: {str(e)[:150]}", flush=True)
    raise ultima


def main() -> None:
    planuri = [(doc, pregateste(doc)) for doc in DOCUMENTE]
    for doc, chunks in planuri:
        lungimi = [len(c["continut"]) for c in chunks]
        print(f"{len(chunks):4d} fragmente, medie {sum(lungimi) // len(lungimi):5d} caractere, max {max(lungimi):5d} | {doc['sursa']}")
        print(f"       exemple: {[c['articol'][:40] for c in chunks[:3]]}")
    print(f"TOTAL: {sum(len(c) for _, c in planuri)} fragmente\n")

    if "--proba" in sys.argv:
        return

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_INGEST_KEY"])

    embedding_cu_reincercari(client, "test conexiune")
    backup_si_sterge(sb)

    rez = sb.table("fiscal_knowledge").select("chunk_index").order("chunk_index", desc=True).limit(1).execute()
    index = (rez.data[0]["chunk_index"] + 1) if rez.data else 0

    esecuri = 0
    for doc, chunks in planuri:
        existente = {r["continut"] for r in toate_randurile(sb, doc["sursa"], "continut")}
        print(f"=== {doc['sursa']} ({len(existente)} deja salvate) ===", flush=True)
        salvate = 0
        for chunk in chunks:
            if chunk["continut"] in existente:
                continue
            try:
                embedding = embedding_cu_reincercari(client, chunk["continut"])
                sb.table("fiscal_knowledge").insert({**chunk, "embedding": embedding, "chunk_index": index}).execute()
            except Exception as e:
                esecuri += 1
                print(f"  ESEC {chunk['articol'][:50]}: {str(e)[:150]}", flush=True)
                continue
            index += 1
            salvate += 1
            if salvate % 25 == 0:
                print(f"  ... {salvate} salvate", flush=True)
            time.sleep(PAUZA)
        print(f"  Gata: {salvate} noi", flush=True)

    total = sb.table("fiscal_knowledge").select("id", count="exact").limit(1).execute().count
    print(f"\nTotal fragmente in baza: {total} | esecuri: {esecuri}")
    if esecuri:
        print("Reporneste scriptul: va incerca doar fragmentele lipsa.")
        return

    print(sb.rpc("recreate_ivfflat_index", {}).execute().data)
    print("=== GATA ===")


if __name__ == "__main__":
    main()
