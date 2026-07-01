"""
Ingestie documente legislative noi in Supabase (fiscal_knowledge).

Proceseaza PDF-uri cu structura standard romaneasca (ART. N - Titlu)
si le adauga in aceeasi baza de date ca Codul Fiscal.

Ruleaza: python ingest_documente_noi.py
"""

import os
import re
import time

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
PAUZA = 4  # secunde intre embeddings (quota Gemini)

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
]

ARTICOL_RE = re.compile(r"\b(?:ART\.?\s*(?:ART\.?\s*)?|Articolul\s+)(\d+\^?\d*)\s*[-–.]?\s*([^\n]*)")
TITLU_RE = re.compile(r"\b(?:TITLUL|CAPITOLUL|SECTIUNEA|SECȚIUNEA)\s+\S+\s*[-–]?\s*([^\n]+)", re.IGNORECASE)


def extrage_text(pdf_path: str) -> str:
    print(f"  Citesc: {pdf_path}")
    reader = PdfReader(pdf_path)
    pagini = []
    for p in reader.pages:
        pagini.append(p.extract_text() or "")
    text = "\n".join(pagini)
    print(f"  {len(text):,} caractere, {len(reader.pages)} pagini")
    return text


def chunking_generic(text: str, sursa: str) -> list[dict]:
    """Imparte textul in bucati pe articole (ART. N)."""
    pozitii = list(ARTICOL_RE.finditer(text))
    if not pozitii:
        print(f"  ATENTIE: niciun articol gasit in {sursa} — ingestie ca bloc unic")
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

        # titlul sectiunii active
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
    """Sub-imparte bucatile prea mari cu suprapunere."""
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
            fragmente.append({
                **chunk,
                "continut": bucata,
                "articol": f"{chunk['articol']} (partea {parte})",
            })
            parte += 1

        if end >= len(continut):
            break
        start = end - SUPRAPUNERE

    return fragmente


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_INGEST_KEY"])

    # aflam chunk_index maxim existent ca sa continuam de acolo
    rez = sb.table("fiscal_knowledge").select("chunk_index").order("chunk_index", desc=True).limit(1).execute()
    index_start = (rez.data[0]["chunk_index"] + 1) if rez.data else 0
    print(f"Chunk index de start: {index_start}\n")

    for doc in DOCUMENTE:
        pdf_path = doc["pdf"]
        sursa = doc["sursa"]

        if not os.path.exists(pdf_path):
            print(f"LIPSA: {pdf_path} — sarit\n")
            continue

        # sari documentele deja complet ingerate
        count_rez = sb.table("fiscal_knowledge").select("id", count="exact").eq("sursa", sursa).limit(1).execute()
        if count_rez.count and count_rez.count > 0:
            print(f"SARIT (deja ingerat {count_rez.count} chunks): {sursa}\n")
            index_start += count_rez.count
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
            chunk_index = index_start + i

            try:
                rezultat = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=chunk["continut"][:18000],
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBED_DIM,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
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
                "chunk_index": chunk_index,
            }).execute()

            reusite += 1
            if reusite % 20 == 0:
                print(f"  ... {reusite}/{len(chunks)} salvate")

            time.sleep(PAUZA)

        index_start += len(chunks)
        print(f"  Finalizat: {reusite}/{len(chunks)} bucati ingerate\n")

    print("=== INGESTIE COMPLETA ===")
    rez = sb.table("fiscal_knowledge").select("id", count="exact").limit(1).execute()
    print(f"Total randuri in fiscal_knowledge: {rez.count}")


if __name__ == "__main__":
    main()
