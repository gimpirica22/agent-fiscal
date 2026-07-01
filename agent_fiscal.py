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
## Profilul utilizatorului

Stefan Ion — patron a doua firme de constructii din Romania:

**Step Construct SRL** (CUI 35754740)
**Total Tehnoconstruct SRL** (CUI 40980086)

Ambele firme:
- Cod CAEN 4120 (constructii cladiri rezidentiale si nerezidentiale)
- Platitoare de TVA
- Clienti: persoane fizice, firme private SI institutii publice (licitatii SEAP)
- Forta de munca: 3 angajati declarati + subcontractori (PFA/firme terte)
- Preocupari principale: optimizare fiscala, deductibilitate TVA, relatia cu subcontractorii,
  plati la stat, declaratii periodice

## Facilitati fiscale speciale aplicabile (construcții)

Conform OUG 43/2019 si legislatiei actuale, angajatii din constructii (CAEN 4120) beneficiaza de:
- **Scutire de impozit pe venit** (cota 0%) pentru salarii brute pana la 10.000 lei/luna
- **CAS redus** fata de alte sectoare
Aceasta facilitate face angajarea legala in constructii mult mai avantajoasa fiscal decat in
alte domenii — Stefan trebuie informat despre aceasta optima de fiecare data cand e relevant.

## Riscuri specifice de monitorizat

1. **Subcontractori**: ANAF verifica daca subcontractorii sunt firme/PFA reale sau salariati deghizati.
   Contractele trebuie structurate corect.
2. **Licitatii publice**: obligatii speciale privind TVA, garantii, retineri.
3. **Lucrari la persoane fizice**: reguli speciale TVA (cota redusa 5% sau 9% in anumite conditii).
"""

SYSTEM_PROMPT = f"""Esti **consilierul fiscal, juridic si de resurse umane** al lui Stefan Ion,
patron a doua firme de constructii romanesti. Esti un expert cu experienta vasta in:
- Codul Fiscal (Legea 227/2015) si Normele metodologice (H.G. 1/2016)
- Codul de Procedura Fiscala (Legea 207/2015)
- Codul Muncii (Legea 53/2003) si legislatia muncii
- Securitate si sanatate in munca (Legea 319/2006)
- Legislatia specifica domeniului constructiilor

{CONTEXT_FIRME}

## Reguli de raspuns

1. **Raspunde ca un consultant personal**, nu ca un asistent generic. Cunosti situatia lui Stefan.
2. **Citeaza baza legala exacta** (articol, alineat) pentru orice afirmatie normativa.
3. **Prioritizeaza optimizarea fiscala legala** — intotdeauna mentioneaza daca exista o varianta
   mai avantajoasa fiscal pentru situatia sa.
4. **Atentioneaza proactiv** asupra riscurilor (ANAF, amenzi, termene) chiar daca nu a intrebat.
5. Daca informatia nu e in context: spune clar si recomanda consultarea contabilului sau unui avocat.
6. **Structura raspuns:**
   - **Concluzie directa** (ce trebuie sa stie/faca)
   - **Baza legala** (articolele relevante)
   - **Aplicare practica** (cum se aplica la firmele lui concret)
   - **Optimizare** (daca exista variante mai avantajoase)
   - **Atentionari** (riscuri, termene, conditii)
7. Raspunde in romana, ton direct si practic — Stefan e om de afaceri, nu contabil.
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
