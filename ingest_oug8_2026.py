"""
Adauga OUG 8/2026 in baza de legi, fara art. 6.

Art. 6 modifica Codul Fiscal, iar Codul Fiscal din baza e deja textul actualizat cu aceste
modificari. Pentru art. 6 se salveaza doar un fragment de trimitere catre articolele modificate.

Rulare:
    python ingest_oug8_2026.py --proba   (nu modifica nimic)
    python ingest_oug8_2026.py
"""

import os
import re
import sys
import time

from ingest_recrutare import (
    PAUZA,
    embedding_cu_reincercari,
    extrage_text,
    sub_imparte,
    toate_randurile,
)

PDF = "OUG 8 din 2026.pdf"
SURSA = "OUG 8/2026 - Relansare economică, bonificații 3% și modificări fiscale"

TITLURI = {
    1: "Obiectul ordonanței și categoriile de măsuri",
    2: "Ajutor de stat pentru investiții cu impact semnificativ (minimum 1 miliard lei)",
    3: "Scheme de ajutor de stat pentru relansare economică (credit fiscal, granturi)",
    4: "Banca de Investiții și Dezvoltare (BID) - finanțare și garantare",
    5: "Investiții ale fondurilor de pensii administrate privat (Legea 411/2004)",
    7: "Bonificația de 3% la impozitul pe profit / impozitul pe veniturile microîntreprinderilor pe 2025",
    8: "Bonificația de 3% la impozitul pe venit al persoanelor fizice pe 2025",
    9: "TVA la încasare - depășirea plafonului de 4.500.000 lei în ianuarie-februarie 2026",
    10: "Intrarea în vigoare și data de aplicare a modificărilor Codului Fiscal",
    11: "Termene pentru ordinele ministrului finanțelor",
    12: "Exim Banca Românească - fond pentru credite la export",
    13: "Programul pentru pregătirea proiectelor în parteneriat public-privat",
    14: "Acorduri-cadru cu instituții financiare internaționale (PPP)",
    15: "Hotărârea de Guvern pentru implementarea Programului PPP",
    16: "Comitetul interministerial pentru parteneriat public-privat (CI3P)",
}

# Rezumat scris dupa textul art. 6; textul complet al modificarilor este in Codul Fiscal din baza.
REZUMAT_ART6 = """OUG 8/2026 art. 6 - Modificări aduse Codului Fiscal (Legea 227/2015)
Textul modificat se găsește în Codul Fiscal actualizat (articolele de mai jos). Pe scurt:
- art. 18^1 alin. (5): comparația impozit pe profit / impozit minim pe cifra de afaceri (nu se scade creditul fiscal pentru cercetare-dezvoltare);
- art. 20^1 (nou): credit fiscal de 10% din cheltuielile eligibile de cercetare-dezvoltare, alternativ deducerii suplimentare de la art. 20;
- art. 22 alin. (9): în 2026, pentru echipamente tehnologice (subgrupa 2.1) și computere scutite de impozit pe profitul reinvestit se poate folosi și amortizarea accelerată;
- art. 23 lit. r) și s): diferența de credit fiscal pentru cercetare-dezvoltare și bonificația acordată de organul fiscal sunt venituri neimpozabile;
- art. 25 alin. (14)-(15): deducere suplimentară de 50% pentru cheltuielile de listare la bursă;
- art. 26 alin. (9^1): rezervele din profitul reinvestit (art. 22 alin. 5) nu pot fi folosite 5 ani pentru capital, distribuire sau acoperirea pierderilor;
- art. 28 alin. (2) lit. b): pragul mijloacelor fixe crește la 5.000 lei (art. 45 alin. 21^3: cele între 2.500 și 5.000 lei existente la 31.12.2025 se amortizează în continuare);
- art. 28 alin. (8^1)-(8^3): amortizare superaccelerată pentru activele noi din subgrupele 2.1 și 2.4 puse în funcțiune în 2026 - maximum 65% în primul an;
- art. 41-42 și 42^9: impozitul pe profit se declară și se plătește anual; plată anticipată pentru trimestrul I până la 25 a lunii următoare; declarația anuală până la 25 iunie (începând cu anul 2026);
- art. 47 alin. (1) lit. c) și art. 52 alin. (1): plafonul microîntreprinderilor scade la 100.000 euro (aplicabil din 2026);
- art. 48: condiția de salariat - suspendări sub 30 de zile și concediu medical de maximum 30 de zile pe an nu scot firma din regimul micro; firmele noi au 90 de zile pentru angajarea salariatului;
- art. 52 alin. (3): dacă singurul salariat pleacă, firma are 30 de zile să angajeze altul;
- art. 53 lit. p): bonificația acordată de organul fiscal nu intră în baza impozabilă la micro;
- art. 54 alin. (1): la plafonul micro se adaugă veniturile din vânzarea a mai mult de un mijloc fix / teren;
- art. 68, 76, 78, 142: deduceri de până la 400 euro/an pentru pensii facultative, ocupaționale, PEPP și pentru cumpărarea de ETF-uri (din martie 2026);
- art. 282 și 324: plafonul TVA la încasare - 5.000.000 lei între 1 martie și 31 decembrie 2026, 5.500.000 lei din 1 ianuarie 2027.
Datele de aplicare sunt la art. 10 din OUG 8/2026."""

ARTICOL_RE = re.compile(r"^[ \t]*Art\.[ \t]*(\d+)[ \t]*-[ \t]*", re.MULTILINE)


def pregateste() -> list[dict]:
    text = extrage_text(PDF).replace(" ", " ")
    # Numerotarea OUG merge 1, 2, 3...; articolele citate in art. 6 (ex. "Art. 201 -") nu se potrivesc
    pozitii, urmator = [], 1
    for m in ARTICOL_RE.finditer(text):
        if int(m.group(1)) == urmator:
            pozitii.append(m)
            urmator += 1
    if urmator - 1 != 16:
        raise ValueError(f"Am gasit {urmator - 1} articole in loc de 16")

    final = text.find("PRIM-MINISTRU")
    chunks = []
    for idx, m in enumerate(pozitii):
        nr = int(m.group(1))
        if nr == 6:
            chunks.append({"continut": REZUMAT_ART6, "articol": "Art. 6 - Modificări aduse Codului Fiscal (rezumat)",
                           "titlu": "", "sursa": SURSA})
            continue
        end = pozitii[idx + 1].start() if idx + 1 < len(pozitii) else final
        bloc = re.sub(r"[ \t]+\n", "\n", text[m.start():end]).strip()
        articol = f"Art. {nr} - {TITLURI[nr]}"
        chunks.extend(sub_imparte({"continut": f"OUG 8/2026 - {articol}\n{bloc}", "articol": articol,
                                   "titlu": "", "sursa": SURSA}))
    return chunks


def main() -> None:
    chunks = pregateste()
    for c in chunks:
        print(f"{len(c['continut']):6d} | {c['articol']}")
    print(f"TOTAL: {len(chunks)} fragmente\n")
    if "--proba" in sys.argv:
        return

    from google import genai
    from supabase import create_client

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = genai.Client(api_key=os.environ["GEMINI_INGEST_KEY"])

    existente = {r["continut"] for r in toate_randurile(sb, SURSA, "continut")}
    rez = sb.table("fiscal_knowledge").select("chunk_index").order("chunk_index", desc=True).limit(1).execute()
    index = (rez.data[0]["chunk_index"] + 1) if rez.data else 0

    esecuri = salvate = 0
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
        time.sleep(PAUZA)

    print(f"Salvate: {salvate} noi | esecuri: {esecuri}")
    if esecuri:
        print("Reporneste scriptul: va incerca doar fragmentele lipsa.")
        return
    print(sb.rpc("recreate_ivfflat_index", {}).execute().data)
    print("=== GATA ===")


if __name__ == "__main__":
    main()
