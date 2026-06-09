"""
Agent fiscal RAG — Codul Fiscal Romania

Fluxul:
1. Primeste intrebarea utilizatorului
2. Genereaza embedding pentru intrebare (Gemini, task RETRIEVAL_QUERY)
3. Cauta top-K bucati relevante in Supabase (match_fiscal_knowledge)
4. Trimite contextul + intrebarea la Gemini 2.5 Flash cu rol de consilier fiscal
5. Returneaza raspunsul structurat

Utilizare:
    python agent_fiscal.py
    python agent_fiscal.py "Ce cheltuieli pot deduce pentru un autoturism?"
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
GEMINI_MODEL = "gemini-3.5-flash"
TOP_K = 6  # numarul de bucati relevante extrase din baza de date

CONTEXT_FIRME = """
Utilizatorul are doua firme romanesti:
- Step Construct SRL (CUI 35754740) — firma de constructii
- Total Tehnoconstruct SRL (CUI 40980086) — firma de constructii

Ambele sunt platitoare de TVA, active in domeniul constructiilor (cod CAEN 4120 - Lucrari de constructii
a cladirilor rezidentiale si nerezidentiale). Raspunsurile trebuie sa fie practice si aplicabile
direct situatiei acestor firme.
"""

SYSTEM_PROMPT = f"""Esti un consilier fiscal expert in legislatia fiscala romaneasca, specializat in
Codul Fiscal (Legea 227/2015) si Normele metodologice de aplicare (H.G. 1/2016).

{CONTEXT_FIRME}

Reguli de raspuns:
1. Bazeaza-te EXCLUSIV pe textele din Codul Fiscal furnizate in context. Nu inventa prevederi.
2. Citeaza articolul exact (ex: "conform Art. 25 alin. (4) lit. l)...").
3. Daca informatia nu se gaseste in context, spune clar: "Nu am gasit prevederi relevante in
   fragmentele disponibile — recomand consultarea unui expert contabil."
4. Structureaza raspunsul astfel:
   - **Concluzie directa** (1-2 propozitii)
   - **Baza legala** (articolele relevante cu citat scurt)
   - **Aplicare practica** pentru firmele de constructii (daca e cazul)
   - **Atentionari** (conditii, exceptii, riscuri fiscale)
5. Raspunde in limba romana, limbaj clar, fara jargon inutil.
"""


def genereaza_embedding_query(client: genai.Client, intrebare: str) -> list[float]:
    rezultat = client.models.embed_content(
        model=EMBED_MODEL,
        contents=intrebare,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM,
            task_type="RETRIEVAL_QUERY",
        ),
    )
    return rezultat.embeddings[0].values


def cauta_context(sb, embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    rezultat = sb.rpc(
        "match_fiscal_knowledge",
        {"query_embedding": embedding, "match_count": top_k},
    ).execute()
    return rezultat.data or []


def formateaza_context(bucati: list[dict]) -> str:
    parti = []
    for i, b in enumerate(bucati, 1):
        sursa = b.get("sursa", "")
        titlu = b.get("titlu", "")
        articol = b.get("articol", "")
        continut = b.get("continut", "")
        similaritate = b.get("similarity", 0)

        antet = f"[Fragment {i} | similaritate: {similaritate:.3f}]"
        meta = " | ".join(filter(None, [sursa, titlu, articol]))
        parti.append(f"{antet}\n{meta}\n{continut}")

    return "\n\n---\n\n".join(parti)


def raspunde(intrebare: str, verbose: bool = True) -> str:
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    if verbose:
        print("Generez embedding pentru intrebare...")
    embedding = genereaza_embedding_query(gemini_client, intrebare)

    if verbose:
        print(f"Caut context relevant in baza de date ({TOP_K} fragmente)...")
    bucati = cauta_context(sb, embedding)

    if not bucati:
        return "Eroare: nu am putut gasi fragmente relevante in baza de date."

    if verbose:
        print(f"Gasit {len(bucati)} fragmente (similaritate maxima: {bucati[0].get('similarity', 0):.3f})")
        print("Generez raspuns cu Gemini 2.5 Flash...\n")

    context = formateaza_context(bucati)

    prompt_complet = (
        f"{SYSTEM_PROMPT}\n\n"
        f"FRAGMENTE DIN CODUL FISCAL:\n\n{context}\n\n"
        f"---\n\nINTREBARE: {intrebare}"
    )

    raspuns = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt_complet,
    )

    return raspuns.text


def main():
    if len(sys.argv) > 1:
        intrebare = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        print("  AGENT FISCAL — Codul Fiscal Romania")
        print("=" * 60)
        print("Firmele tale: Step Construct SRL | Total Tehnoconstruct SRL")
        print("Scrie 'exit' pentru a iesi.\n")
        intrebare = input("Intrebarea ta: ").strip()
        if intrebare.lower() in ("exit", "quit", ""):
            return

    print()
    raspuns = raspunde(intrebare)
    print(raspuns)
    print()

    # Mod interactiv continuu daca nu s-a dat argument din linie de comanda
    if len(sys.argv) == 1:
        while True:
            print("-" * 60)
            intrebare = input("Alta intrebare (sau 'exit'): ").strip()
            if intrebare.lower() in ("exit", "quit", ""):
                break
            print()
            raspuns = raspunde(intrebare, verbose=False)
            print(raspuns)
            print()


if __name__ == "__main__":
    main()
