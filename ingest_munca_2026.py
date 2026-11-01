"""
Actualizeaza legislatia muncii din baza agentului, fara sa atinga legile fiscale.

- incarca OUG 32/2026 (a inlocuit Titlul I din OG 25/2014)
- sterge OG 25/2014
- reincarca Codul Muncii cu articolele intregi (notele de modificare nu le mai rup)

Embedding-urile se calculeaza si se salveaza local INAINTE de orice stergere, iar fragmentele
sterse se salveaza in backup_fragmente_*.json. Scriptul poate fi repornit.

Rulare:
    python ingest_munca_2026.py --proba   (nu modifica nimic)
    python ingest_munca_2026.py
"""

import json
import os
import re
import sys
import time
from datetime import date

from google import genai
from pypdf import PdfReader
from supabase import create_client

from ingest_recrutare import PAUZA, embedding_cu_reincercari, sub_imparte, toate_randurile

SURSA_CM = "Legea 53/2003 - Codul Muncii"
SURSA_OUG = "OUG 32/2026 - Accesul străinilor pe piața muncii (a înlocuit Titlul I din OG 25/2014)"
SURSA_OG25 = "OG 25/2014 - Încadrarea în muncă și detașarea străinilor"
PDF_CM = "CODUL MUNCII din 24 ianuarie 2003.pdf"
PDF_OUG = "OUG 32 din 2026.pdf"
CACHE = "cache_embeddinguri_munca.json"

# Doar randurile care contin exclusiv "Articolul N" sunt titluri de articol. Randurile care incep
# asa dar continua ("Articolul 27 , Capitolul I ...") sunt note de modificare.
TITLU_ARTICOL = re.compile(r"^[ \t]*Articolul[ \t]+(\d+(?:\^\d+)?)[ \t]*$", re.MULTILINE)


def text_pdf(cale: str, mod: str = "plain") -> str:
    return "\n".join(p.extract_text(extraction_mode=mod) or "" for p in PdfReader(cale).pages)


def _rupe(text: str, pozitii: list[re.Match]) -> list[tuple[str, int, str]]:
    blocuri = []
    for i, m in enumerate(pozitii):
        sfarsit = pozitii[i + 1].start() if i + 1 < len(pozitii) else len(text)
        blocuri.append((m.group(1), m.start(), text[m.start():sfarsit].strip()))
    return blocuri


def fragmente_codul_muncii() -> list[dict]:
    text = text_pdf(PDF_CM)
    pozitii, ultimul = [], 0
    for m in TITLU_ARTICOL.finditer(text):
        numar = int(m.group(1).split("^")[0])
        if numar < ultimul:  # articol citat in interiorul unei note, nu titlu real
            continue
        pozitii.append(m)
        ultimul = numar
    fragmente = []
    for numar, _, bloc in _rupe(text, pozitii):
        fragmente += sub_imparte({"continut": bloc, "articol": f"Art. {numar}", "titlu": "", "sursa": SURSA_CM})
    return fragmente


def fragmente_oug32() -> list[dict]:
    text = text_pdf(PDF_OUG, "layout")
    text = re.sub(r"(?m)^[ \t]*Pagina \d+ din \d+[ \t]*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Titlul II-IV contin articole din alte legi, in forma modificata de OUG 32/2026.
    limite = {t: re.search(rf"(?m)^ ?Titlul {t} ?$", text).start() for t in ("I", "II", "III", "IV", "V")}
    articole_proprii = {"45", "46", "47"}

    fragmente = []
    for numar, pozitie, bloc in _rupe(text, list(TITLU_ARTICOL.finditer(text, limite["I"]))):
        if limite["II"] < pozitie < limite["III"] and numar not in articole_proprii:
            eticheta, titlu = f"Art. {numar} din OUG 194/2002, forma din OUG 32/2026", "Regimul străinilor"
        elif limite["III"] < pozitie < limite["IV"] and numar not in articole_proprii:
            eticheta, titlu = f"Art. {numar} din Codul muncii, forma din OUG 32/2026", "Codul muncii"
        elif limite["IV"] < pozitie < limite["V"] and numar not in articole_proprii:
            eticheta, titlu = f"Art. {numar} din Legea 198/2008, forma din OUG 32/2026", "Taxe consulare"
        elif pozitie > limite["V"]:
            eticheta, titlu = f"Art. {numar}", "Dispoziții finale și tranzitorii"
        else:
            eticheta, titlu = f"Art. {numar}", "Angajatori, agenții de plasare, obligații, contravenții"
        fragmente += sub_imparte({"continut": bloc, "articol": eticheta, "titlu": titlu, "sursa": SURSA_OUG})
    return fragmente


def cheie(f: dict) -> str:
    return f"{f['articol']}|{f['continut']}"


def calculeaza_embeddinguri(fragmente: list[dict]) -> dict:
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    client = genai.Client(api_key=os.environ["GEMINI_INGEST_KEY"])
    lipsa = [f for f in fragmente if cheie(f) not in cache]
    print(f"Embedding-uri: {len(cache)} salvate, {len(lipsa)} de calculat", flush=True)
    for i, f in enumerate(lipsa, 1):
        cache[cheie(f)] = embedding_cu_reincercari(client, f["continut"])
        if i % 25 == 0 or i == len(lipsa):
            with open(CACHE, "w", encoding="utf-8") as g:
                json.dump(cache, g)
            print(f"  ... {i}/{len(lipsa)}", flush=True)
        time.sleep(PAUZA)
    return cache


def salveaza_backup(randuri: list[dict]) -> None:
    fisier = f"backup_fragmente_{date.today().isoformat()}.json"
    salvate = json.load(open(fisier, encoding="utf-8")) if os.path.exists(fisier) else []
    with open(fisier, "w", encoding="utf-8") as g:
        json.dump(salvate + randuri, g, ensure_ascii=False)


def sterge(sb, randuri: list[dict], eticheta: str) -> None:
    if not randuri:
        return
    salveaza_backup(randuri)
    ids = [r["id"] for r in randuri]
    for i in range(0, len(ids), 100):
        sb.table("fiscal_knowledge").delete().in_("id", ids[i:i + 100]).execute()
    print(f"Sterse {len(ids)} fragmente vechi: {eticheta}", flush=True)


def main() -> None:
    cm, oug = fragmente_codul_muncii(), fragmente_oug32()
    for nume, lista in (("Codul Muncii", cm), ("OUG 32/2026", oug)):
        lungimi = [len(f["continut"]) for f in lista]
        print(f"{nume}: {len(lista)} fragmente, medie {sum(lungimi) // len(lungimi)}, max {max(lungimi)}")
        print("  etichete:", [f["articol"] for f in lista[:4]], "...", [f["articol"] for f in lista[-3:]])
    if "--proba" in sys.argv:
        for f in oug:
            print(" ", f["articol"], "|", len(f["continut"]))
        return

    fragmente = cm + oug
    cache = calculeaza_embeddinguri(fragmente)

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    noi = {cheie(f) for f in fragmente}
    vechi_cm = toate_randurile(sb, SURSA_CM, "*")
    sterge(sb, toate_randurile(sb, SURSA_OG25, "*"), SURSA_OG25)
    sterge(sb, [r for r in vechi_cm if cheie(r) not in noi], SURSA_CM)

    existente = {cheie(r) for r in vechi_cm} | {cheie(r) for r in toate_randurile(sb, SURSA_OUG, "articol,continut")}
    rez = sb.table("fiscal_knowledge").select("chunk_index").order("chunk_index", desc=True).limit(1).execute()
    index = (rez.data[0]["chunk_index"] + 1) if rez.data else 0

    de_adaugat = [f for f in fragmente if cheie(f) not in existente]
    for i, f in enumerate(de_adaugat, 1):
        sb.table("fiscal_knowledge").insert({**f, "embedding": cache[cheie(f)], "chunk_index": index}).execute()
        index += 1
        if i % 50 == 0:
            print(f"  ... {i}/{len(de_adaugat)} salvate", flush=True)

    total = sb.table("fiscal_knowledge").select("id", count="exact").limit(1).execute().count
    print(f"Adaugate: {len(de_adaugat)} | Total fragmente in baza: {total}")
    print(sb.rpc("recreate_ivfflat_index", {}).execute().data)
    print("=== GATA ===")


if __name__ == "__main__":
    main()
