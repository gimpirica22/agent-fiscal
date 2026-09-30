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
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from supabase import create_client

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
GEMINI_MODEL = "gemini-3.5-flash"
TOP_K = 9  # numarul de bucati relevante extrase din baza de date
MAX_HISTORY = 5  # numarul de schimburi (intrebare+raspuns) pastrate in memorie
RETRY_CODES = {503, 429}  # model suprasolicitat / cota depasita — merita reincercat
RETRY_DELAYS = [3, 8, 15]  # secunde intre reincercari

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

## A treia firma: SDI Capital Group SRL (infiintata recent, inca neconectata operational)

**SDI Capital Group SRL** (CUI 55528535, Nr. Reg. Com. J2026052709007)
- Infiintata 05.08.2026, inmatriculata 03.09.2026 — firma foarte noua
- Cod CAEN 6421 (activitati de holding), cu activitati secundare IT, imobiliare,
  consultanta in afaceri si leasing echipamente pentru constructii
- Asociati: Stefan Ion 50% (administrator, mandat pana in 2046) si Stefan Daria Ioana 50%
- Sediu: Ploiesti, Str. Tatarani, Nr. 9, Prahova

IMPORTANT: la data actualizarii acestui profil, SDI Capital Group NU detine inca parti
sociale la Step Construct SRL sau Total Tehnoconstruct SRL. Cesiunea de parti sociale prin
care SDI ar deveni asociat la cele doua firme de constructii este doar planificata, nu s-a
realizat. NU presupune si NU afirma o relatie de holding activa (detinere, consolidare,
scutire de dividende) intre SDI si cele doua firme pana cand aceasta informatie nu este
actualizata explicit aici.

## Atentie: facilitati fiscale constructii — verifica mereu in context

Facilitatile fiscale pentru sectorul constructii (impozit pe venit, CAS) s-au modificat frecvent
in ultimii ani (OUG 114/2018, OUG 43/2019 si modificari ulterioare). NU presupune ca regulile
vechi mai sunt valabile — verifica intotdeauna starea curenta (activa/abrogata) direct in
fragmentele din context inainte sa afirmi ceva despre ele.

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
8. **Verifica starea de valabilitate a fiecarei prevederi** inainte sa o citezi: daca un fragment
   contine cuvantul ABROGAT (sau A B R O G A T) langa un alineat/punct/litera, acea prevedere
   NU mai este in vigoare — nu o cita niciodata ca fiind aplicabila, chiar daca restul articolului
   sau cunostintele tale generale sugereaza contrariul. Daca prevederea cautata de utilizator
   e abrogata, spune-i explicit asta si cauta daca exista o prevedere curenta care a inlocuit-o.
9. **Nu inventa baza legala — nici macar in detalii secundare sau mentionate in treacat.**
   Citeaza doar articole, ordonante, legi, cote si plafoane care apar LITERAL in fragmentele
   primite ca context. Aceasta regula se aplica INTEGRAL raspunsului, nu doar concluziei
   principale — inclusiv exemplelor, comparatiilor si mentiunilor incidentale (ex: "cota de TVA
   la achizitie e X%" mentionata in treacat intr-un raspuns despre altceva). Daca un detaliu
   (cota exacta, act normativ, prag valoric) nu apare in fragmentele primite, NU il completa din
   cunostintele tale generale de antrenare — omite mentiunea sau spune clar ca nu ai gasit
   detaliul in baza de date si recomanda verificarea cu contabilul, in loc sa presupui sau sa
   aproximezi.
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


def genereaza_cu_retry(gemini_client: genai.Client, contents: list, verbose: bool = True):
    """Apeleaza generate_content cu reincercare pentru erorile temporare (503/429)."""
    for incercare, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            if verbose:
                print(f"Model suprasolicitat, reincerc in {delay}s...")
            time.sleep(delay)
        try:
            return gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
        except errors.APIError as e:
            if e.code not in RETRY_CODES or incercare == len(RETRY_DELAYS):
                raise
    raise RuntimeError("neatins")  # pentru linter — bucla de mai sus mereu returneaza sau arunca


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

    raspuns = genereaza_cu_retry(gemini_client, contents, verbose=verbose)

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
