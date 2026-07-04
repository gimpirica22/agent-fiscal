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
MAX_HISTORY = 5  # numarul de schimburi (intrebare+raspuns) pastrate in memorie

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
- Codul Fiscal (Legea 227/2015) actualizat cu toate modificarile pana la OUG 8/2026
- Normele metodologice de aplicare a Codului Fiscal (H.G. 1/2016, actualizat HG 602/2025)
- Codul de Procedura Fiscala (Legea 207/2015)
- Codul Muncii (Legea 53/2003) si legislatia muncii
- Securitate si sanatate in munca (Legea 319/2006) si Normele metodologice (HG 1425/2006)
- REVISAL — Registrul General de Evidenta a Salariatilor (HG 500/2011)
- Legea Societatilor Comerciale (Legea 31/1990) — capital social, asociati, dividende, administrare
- Legea plafoanelor de numerar (Legea 70/2015) — reguli cash, plafoane incasari/plati
- Legea Registrului Comertului (Legea 265/2022) — infiintare firme, procedura ONRC, modificari acte
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


def raspunde(intrebare: str, history: list[dict] | None = None, verbose: bool = True) -> tuple[str, list[dict]]:
    """
    Returneaza (text_raspuns, history_actualizat).
    history: lista de dict {"role": "user"|"model", "text": "..."}
    """
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    if verbose:
        print("Generez embedding pentru intrebare...")
    embedding = genereaza_embedding_query(gemini_client, intrebare)

    if verbose:
        print(f"Caut context relevant in baza de date ({TOP_K} fragmente)...")
    bucati = cauta_context(sb, embedding)

    if not bucati:
        return "Eroare: nu am putut gasi fragmente relevante in baza de date.", history or []

    if verbose:
        print(f"Gasit {len(bucati)} fragmente (similaritate maxima: {bucati[0].get('similarity', 0):.3f})")
        print("Generez raspuns cu Gemini Flash...\n")

    context = formateaza_context(bucati)

    # Construim conversatia multi-turn cu istoricul anterior
    contents = []
    for turn in (history or []):
        contents.append(
            types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        )
    # Mesajul curent include contextul RAG proaspat
    mesaj_curent = f"FRAGMENTE LEGISLATIVE RELEVANTE:\n\n{context}\n\n---\n\nINTREBARE: {intrebare}"
    contents.append(types.Content(role="user", parts=[types.Part(text=mesaj_curent)]))

    raspuns = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    text = raspuns.text

    # Actualizam istoricul cu intrebarea curata (fara contextul RAG) si raspunsul
    history_nou = list(history or []) + [
        {"role": "user", "text": intrebare},
        {"role": "model", "text": text},
    ]
    # Pastram doar ultimele MAX_HISTORY schimburi (fiecare schimb = 2 intrari)
    if len(history_nou) > MAX_HISTORY * 2:
        history_nou = history_nou[-MAX_HISTORY * 2:]

    return text, history_nou


def main():
    history: list[dict] = []

    if len(sys.argv) > 1:
        intrebare = " ".join(sys.argv[1:])
        print()
        raspuns, _ = raspunde(intrebare)
        print(raspuns)
        print()
        return

    print("=" * 60)
    print("  AGENT FISCAL — Codul Fiscal Romania")
    print("=" * 60)
    print("Firmele tale: Step Construct SRL | Total Tehnoconstruct SRL")
    print("Scrie 'nou' pentru a reseta conversatia, 'exit' pentru a iesi.\n")

    while True:
        intrebare = input("Intrebarea ta: ").strip()
        if intrebare.lower() in ("exit", "quit", ""):
            break
        if intrebare.lower() == "nou":
            history = []
            print("Conversatie resetata.\n")
            continue
        if not intrebare:
            continue
        print()
        raspuns, history = raspunde(intrebare, history=history, verbose=False)
        print(raspuns)
        print(f"\n[Memorie: {len(history)//2} schimburi]\n")


if __name__ == "__main__":
    main()
