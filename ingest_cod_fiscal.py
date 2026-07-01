"""
Ingestie Cod Fiscal -> Supabase (pgvector)

Documentul PDF contine doua sectiuni cu structuri diferite:
  1. Codul Fiscal propriu-zis (Legea 227/2015): organizat pe TITLUL -> ART. N - Titlu
  2. Normele metodologice (HG 1/2016): organizat pe TITLUL -> CAPITOLUL -> SECTIUNEA -> paragrafe

Pasi:
1. Extrage textul din PDF
2. Separa cele doua sectiuni si le imparte fiecare cu logica proprie
3. Genereaza embeddings cu Gemini pentru fiecare bucata
4. Insereaza in Supabase (tabel fiscal_knowledge)

Ruleaza: python ingest_cod_fiscal.py
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

PDF_PATH = "codul fiscal.pdf"
SURSA_COD = "Legea 227/2015 - Codul Fiscal (actualizat 2026)"
SURSA_NORME = "H.G. 1/2016 - Norme metodologice de aplicare a Codului Fiscal"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20  # cate randuri trimitem deodata catre Supabase

MARCAJ_NORME = "NORME METODOLOGICE"
MAX_CARACTERE_CHUNK = 6000
SUPRAPUNERE = 500

ARTICOL_RE = re.compile(r"ART\.\s*(?:ART\.\s*)?(\d+\^?\d*)\s*[-–]\s*([^\n]+)")
TITLU_RE = re.compile(r"\bTITLUL\s+([IVX]+1?)\s*[-–]\s*([^\n(]+)")
CAPITOL_RE = re.compile(r"\bCAPITOLUL\s+([IVX]+1?)\s*[-–]\s*([^\n]+)")
SECTIUNE_RE = re.compile(r"\bSEC[ȚT]IUNEA\s+(?:a\s+)?[\dIVXa-z]+(?:-\w+)?\s*[-–]\s*([^\n]+)")


def extrage_text(pdf_path: str) -> str:
    print(f"Citesc PDF: {pdf_path} ...")
    reader = PdfReader(pdf_path)
    parti = []
    for i, pagina in enumerate(reader.pages):
        parti.append(pagina.extract_text() or "")
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(reader.pages)} pagini citite")
    text = "\n".join(parti)
    print(f"Text extras: {len(text):,} caractere din {len(reader.pages)} pagini")
    return text


def _gaseste_titlu_curent(titluri_pozitii, pozitie):
    """Returneaza eticheta titlului activ la o anumita pozitie in text."""
    curent = ""
    for poz, eticheta in titluri_pozitii:
        if poz <= pozitie:
            curent = eticheta
        else:
            break
    return curent


def imparte_cod_fiscal(text_cod: str):
    """
    Imparte corpul Codului Fiscal in bucati pe articole (ART. N - Titlu),
    pastrand contextul TITLULUI din care face parte fiecare articol.
    """
    titluri_pozitii = [
        (m.start(), f"TITLUL {m.group(1)} - {m.group(2).strip()}")
        for m in TITLU_RE.finditer(text_cod)
    ]

    pozitii = list(ARTICOL_RE.finditer(text_cod))
    chunks = []

    for idx, m in enumerate(pozitii):
        start = m.start()
        end = pozitii[idx + 1].start() if idx + 1 < len(pozitii) else len(text_cod)
        bloc = text_cod[start:end].strip()
        if len(bloc) < 20:
            continue

        chunks.append({
            "continut": bloc,
            "articol": f"Art. {m.group(1)} - {m.group(2).strip()}",
            "titlu": _gaseste_titlu_curent(titluri_pozitii, start),
            "sursa": SURSA_COD,
        })

    return chunks


def imparte_norme_metodologice(text_norme: str):
    """
    Imparte anexa de Norme metodologice in bucati pe SECTIUNI (sau CAPITOLE,
    cand un capitol nu are sectiuni proprii), pastrand contextul TITLU/CAPITOL.
    """
    titluri_pozitii = [
        (m.start(), f"TITLUL {m.group(1)} - {m.group(2).strip()}")
        for m in TITLU_RE.finditer(text_norme)
    ]
    capitole_pozitii = [
        (m.start(), f"CAPITOLUL {m.group(1)} - {m.group(2).strip()}")
        for m in CAPITOL_RE.finditer(text_norme)
    ]

    # punctele de tăiere sunt fie SECTIUNEA, fie CAPITOLUL (cand nu are sectiuni)
    taieturi = []
    for m in SECTIUNE_RE.finditer(text_norme):
        taieturi.append((m.start(), "sectiune", m.group(1).strip()))
    for m in CAPITOL_RE.finditer(text_norme):
        taieturi.append((m.start(), "capitol", m.group(2).strip()))
    taieturi.sort(key=lambda t: t[0])

    # pastram doar taieturile de capitol care nu sunt urmate imediat de o sectiune
    # (adica acel capitol nu se subdivide -> capitolul intreg devine o bucata)
    filtrate = []
    for i, (poz, tip, nume) in enumerate(taieturi):
        if tip == "capitol":
            urmatoarea = taieturi[i + 1] if i + 1 < len(taieturi) else None
            if urmatoarea and urmatoarea[1] == "sectiune" and urmatoarea[0] - poz < 2000:
                continue  # capitolul are sectiuni proprii -> nu il folosim ca taietura
        filtrate.append((poz, tip, nume))

    chunks = []
    for idx, (poz, tip, nume) in enumerate(filtrate):
        start = poz
        end = filtrate[idx + 1][0] if idx + 1 < len(filtrate) else len(text_norme)
        bloc = text_norme[start:end].strip()
        if len(bloc) < 20:
            continue

        eticheta_capitol = _gaseste_titlu_curent(capitole_pozitii, start)
        chunks.append({
            "continut": bloc,
            "articol": ("Sectiunea: " if tip == "sectiune" else "Capitolul: ") + nume,
            "titlu": _gaseste_titlu_curent(titluri_pozitii, start) +
                     (f" / {eticheta_capitol}" if eticheta_capitol and tip == "sectiune" else ""),
            "sursa": SURSA_NORME,
        })

    return chunks


def divizeaza_daca_e_lung(chunk):
    """Sub-imparte o bucata prea mare in fragmente cu suprapunere, taind la limite de paragraf."""
    continut = chunk["continut"]
    if len(continut) <= MAX_CARACTERE_CHUNK:
        return [chunk]

    fragmente = []
    start = 0
    parte = 1
    while start < len(continut):
        end = min(start + MAX_CARACTERE_CHUNK, len(continut))
        if end < len(continut):
            taietura = continut.rfind("\n", start + 1000, end)
            if taietura == -1:
                taietura = continut.rfind(". ", start + 1000, end)
            if taietura != -1:
                end = taietura + 1

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


def imparte_document(text: str):
    marcaj_idx = text.find(MARCAJ_NORME)
    prima_aparitie_art1 = text.find("ART. 1 - Scopul")

    # antetul real "TITLUL I - Dispozitii generale" apare chiar inainte de ART. 1
    # (cuprinsul de la inceputul documentului contine si el "TITLUL I", de aceea
    # luam ultima aparitie dinaintea lui ART. 1, nu prima)
    titluri_inainte = list(TITLU_RE.finditer(text[:prima_aparitie_art1 + 50]))
    inceput_cod_idx = titluri_inainte[-1].start() if titluri_inainte else prima_aparitie_art1

    text_cod = text[inceput_cod_idx:marcaj_idx]
    text_norme = text[marcaj_idx:]

    print(f"Sectiunea Cod Fiscal: {len(text_cod):,} caractere")
    print(f"Sectiunea Norme metodologice: {len(text_norme):,} caractere")

    chunks_cod = imparte_cod_fiscal(text_cod)
    chunks_norme = imparte_norme_metodologice(text_norme)

    print(f"Articole din Codul Fiscal: {len(chunks_cod)}")
    print(f"Sectiuni/capitole din Normele metodologice: {len(chunks_norme)}")

    chunks_finale = []
    for c in chunks_cod + chunks_norme:
        chunks_finale.extend(divizeaza_daca_e_lung(c))

    print(f"Total bucati dupa sub-impartirea celor prea mari: {len(chunks_finale)}")
    return chunks_finale


def genereaza_embedding(client: genai.Client, text: str) -> list[float]:
    rezultat = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM,
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    return rezultat.embeddings[0].values


def main():
    gemini_key = os.environ["GEMINI_API_KEY"]
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=gemini_key)

    text = extrage_text(PDF_PATH)
    chunks = imparte_document(text)
    print(f"Total bucati de procesat: {len(chunks)}")

    randuri = []
    for i, chunk in enumerate(chunks):
        # textele foarte lungi se taie la limita acceptata de modelul de embeddings
        continut_pt_embedding = chunk["continut"][:18000]
        try:
            embedding = genereaza_embedding(client, continut_pt_embedding)
        except Exception as e:
            print(f"  [eroare embedding la '{chunk['articol']}']: {e}")
            time.sleep(5)
            continue

        randuri.append({
            "continut": chunk["continut"],
            "embedding": embedding,
            "sursa": chunk["sursa"],
            "titlu": chunk["titlu"],
            "articol": chunk["articol"],
            "chunk_index": i,
        })

        if len(randuri) >= BATCH_SIZE:
            sb.table("fiscal_knowledge").insert(randuri).execute()
            print(f"  ... {i + 1}/{len(chunks)} procesate si salvate")
            randuri = []

        time.sleep(0.3)  # cruta rate-limit-ul Gemini

    if randuri:
        sb.table("fiscal_knowledge").insert(randuri).execute()
        print(f"  ... {len(chunks)}/{len(chunks)} procesate si salvate")

    print("Ingestie finalizata cu succes.")


if __name__ == "__main__":
    main()
