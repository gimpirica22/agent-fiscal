"""
Ingestie DOAR pentru documentele care lipsesc din Supabase.
Verifica inainte ce surse exista si sare peste cele deja ingerate.

Ruleaza: python ingest_lipsa.py
"""

import os
import re
import sys
import time

# Fix encoding pentru terminal Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import create_client
from google import genai
from google.genai import types

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
MAX_CHUNK = 6000
SUPRAPUNERE = 500
PAUZA = 4

DOCUMENTE = [
    {
        "pdf": "Legea nr 207 din 2015 privind Codul de procedură fiscală.pdf",
        "sursa": "Legea 207/2015 - Codul de Procedură Fiscală",
    },
    {
        "pdf": "CODUL MUNCII din 24 ianuarie 2003.pdf",
        "sursa": "Legea 53/2003 - Codul Muncii",
    },
    {
        "pdf": "HOTĂRÂRE nr. 500 din 18 mai 2011 privind registrul general de evidență a salariaților.pdf",
        "sursa": "HG 500/2011 - Registrul General de Evidență a Salariaților (REVISAL)",
    },
    {
        "pdf": "LEGE nr. 319 din 14 iulie 2006 a securității și sănătății în muncă.pdf",
        "sursa": "Legea 319/2006 - Securitatea și Sănătatea în Muncă (SSM)",
    },
    {
        "pdf": "HOTĂRÂRE nr. 1425 din 11 octombrie 2006.pdf",
        "sursa": "HG 1425/2006 - Norme metodologice SSM",
    },
    {
        "pdf": "LEGE nr. 31 din 16 noiembrie 1990.pdf",
        "sursa": "Legea 31/1990 - Legea Societăților Comerciale",
    },
    {
        "pdf": "LEGEA nr. 70 din 2 aprilie 2015.pdf",
        "sursa": "Legea 70/2015 - Plafoane de Numerar",
    },
]

ARTICOL_RE = re.compile(r"\b(?:ART\.?\s*(?:ART\.?\s*)?|Articolul\s+)(\d+\^?\d*)\s*[-–.]?\s*([^\n]*)")
TITLU_RE = re.compile(r"\b(?:TITLUL|CAPITOLUL|SECTIUNEA|SECȚIUNEA)\s+\S+\s*[-–]?\s*([^\n]+)", re.IGNORECASE)


def sursa_existenta(sb, sursa: str) -> bool:
    """Verifica daca o sursa specifica exista in Supabase (query exact, nu paginated)."""
    rez = sb.table("fiscal_knowledge").select("id", count="exact").eq("sursa", sursa).limit(1).execute()
    return (rez.count or 0) > 0


def extrage_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    print(f"  {len(text):,} caractere, {len(reader.pages)} pagini")
    return text


def chunking_generic(text: str, sursa: str) -> list[dict]:
    pozitii = list(ARTICOL_RE.finditer(text))
    if not pozitii:
        print(f"  ATENTIE: niciun articol gasit — ingestie ca bloc unic")
        return [{"continut": text[:MAX_CHUNK], "articol": "Document complet", "titlu": "", "sursa": sursa}]

    chunks = []
    for idx, m in enumerate(pozitii):
        start = m.start()
        end = pozitii[idx + 1].start() if idx + 1 < len(pozitii) else len(text)
        bloc = text[start:end].strip()
        if len(bloc) < 20:
            continue
        nr_art = m.group(1)
        titlu_art = m.group(2).strip()[:120] if m.group(2) else ""
        titlu_sectiune = ""
        for tm in TITLU_RE.finditer(text[:start]):
            titlu_sectiune = tm.group(1).strip()[:100]
        chunks.append({
            "continut": bloc,
            "articol": f"Art. {nr_art}" + (f" - {titlu_art}" if titlu_art else ""),
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
            if taietura == -1:
                taietura = end
            end = taietura
        bucata = continut[start:end].strip()
        if bucata:
            fragmente.append({**chunk, "continut": bucata, "articol": f"{chunk['articol']} (partea {parte})"})
            parte += 1
        if end >= len(continut):
            break
        start = end - SUPRAPUNERE
    return fragmente


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_INGEST_KEY"])

    rez = sb.table("fiscal_knowledge").select("chunk_index").order("chunk_index", desc=True).limit(1).execute()
    index_start = (rez.data[0]["chunk_index"] + 1) if rez.data else 0
    print(f"Chunk index de start: {index_start}\n")

    for doc in DOCUMENTE:
        sursa = doc["sursa"]
        pdf_path = doc["pdf"]

        if sursa_existenta(sb, sursa):
            print(f"SARIT (deja ingerat): {sursa}\n")
            continue

        if not os.path.exists(pdf_path):
            print(f"LIPSA PDF: {pdf_path}\n")
            continue

        print(f"=== {sursa} ===")
        text = extrage_text(pdf_path)

        chunks_brute = chunking_generic(text, sursa)
        chunks = []
        for c in chunks_brute:
            chunks.extend(sub_imparte(c))
        print(f"  {len(chunks)} bucati de procesat")

        reusite = 0
        for i, chunk in enumerate(chunks):
            try:
                rezultat = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=chunk["continut"][:18000],
                    config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM, task_type="RETRIEVAL_DOCUMENT"),
                )
                embedding = rezultat.embeddings[0].values
            except Exception as e:
                print(f"  [EROARE embedding {i}]: {e}")
                time.sleep(10)
                continue

            sb.table("fiscal_knowledge").insert({
                "continut": chunk["continut"],
                "embedding": embedding,
                "sursa": chunk["sursa"],
                "titlu": chunk["titlu"],
                "articol": chunk["articol"],
                "chunk_index": index_start + i,
            }).execute()

            reusite += 1
            if reusite % 10 == 0:
                print(f"  ... {reusite}/{len(chunks)} salvate")
            time.sleep(PAUZA)

        index_start += len(chunks)
        print(f"  Finalizat: {reusite}/{len(chunks)} bucati ingerate\n")

    print("=== GATA ===")
    rez = sb.table("fiscal_knowledge").select("id", count="exact").limit(1).execute()
    print(f"Total randuri in fiscal_knowledge: {rez.count}")


if __name__ == "__main__":
    main()
