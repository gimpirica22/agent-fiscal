"""
Citeste balante de verificare exportate din SAGA (PDF), calculeaza indicatorii si le salveaza in Supabase.

Indicatorii se calculeaza aici, in Python, din conturi: modelul AI primeste cifrele gata calculate
si nu le estimeaza el.

Rulare (test local):
    python balanta_saga.py fisier.pdf            (citeste si afiseaza indicatorii)
    python balanta_saga.py fisier.pdf --salveaza (si salveaza in Supabase)
"""

import io
import os
import re
import sys
from dataclasses import dataclass, field

from pypdf import PdfReader

NUMAR = r"-?\d{1,3}(?: \d{3})*\.\d{2}"
CUI_RE = re.compile(r"c\.f\.\s*(?:RO)?\s*(\d{2,10})")
FIRMA_RE = re.compile(r"S\.C\.\s+(.+?)\s+c\.f\.")
PERIOADA_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*--\s*(\d{2})\.(\d{2})\.(\d{4})")
COLOANE = ["si_d", "si_c", "rp_d", "rp_c", "tr_d", "tr_c", "sf_d", "sf_c"]


def numere(text: str) -> list[float]:
    return [float(n.replace(" ", "")) for n in re.findall(NUMAR, text)]


@dataclass
class Balanta:
    firma: str
    cui: str
    data_inceput: str  # YYYY-MM-DD
    data_sfarsit: str
    conturi: list[dict] = field(default_factory=list)

    def cont(self, prefix: str) -> list[dict]:
        return [c for c in self.conturi if c["cont"].startswith(prefix)]

    def suma(self, prefix: str, coloana: str) -> float:
        return round(sum(c[coloana] for c in self.cont(prefix)), 2)

    def sold(self, prefix: str) -> float:
        """Sold final net: pozitiv = debitor, negativ = creditor."""
        return round(self.suma(prefix, "sf_d") - self.suma(prefix, "sf_c"), 2)


def citeste_pdf(sursa) -> Balanta:
    """sursa: cale catre fisier sau bytes."""
    if isinstance(sursa, (bytes, bytearray)):
        sursa = io.BytesIO(sursa)
    text = "\n".join(p.extract_text(extraction_mode="layout") or "" for p in PdfReader(sursa).pages)
    text = text.replace("\xa0", " ")

    firma = FIRMA_RE.search(text)
    cui = CUI_RE.search(text)
    perioada = PERIOADA_RE.search(text)
    if not (cui and perioada):
        raise ValueError("Nu pare o balanta de verificare SAGA (lipsesc CUI-ul sau perioada).")
    z1, l1, a1, z2, l2, a2 = perioada.groups()
    bal = Balanta(
        firma=(firma.group(1).strip() if firma else "").replace("S.R.L.", "SRL"),
        cui=cui.group(1),
        data_inceput=f"{a1}-{l1}-{z1}",
        data_sfarsit=f"{a2}-{l2}-{z2}",
    )

    totaluri_clase: dict[str, list[float]] = {}
    totaluri_generale = None
    for linie in text.split("\n"):
        # In modul "layout" coloanele sunt separate de cel putin 2 spatii; in sume, miile au un spatiu
        bucati = [b for b in re.split(r"\s{2,}", linie.strip()) if b]
        if not bucati:
            continue
        sume = [b for b in bucati if re.fullmatch(NUMAR, b)]
        if bucati[0] == "Total sume clasa" and len(sume) == 8:
            totaluri_clase[bucati[1]] = numere(" ".join(sume))
        elif bucati[0] == "Totaluri:" and len(sume) == 8:
            totaluri_generale = numere(" ".join(sume))
        elif re.fullmatch(r"\d{3,6}", bucati[0]) and len(bucati) >= 10 and all(
                re.fullmatch(NUMAR, b) for b in bucati[-8:]):
            valori = [float(b.replace(" ", "")) for b in bucati[-8:]]
            denumire = " ".join(bucati[1:-8])  # unele denumiri contin doua spatii la rand
            bal.conturi.append({"cont": bucati[0], "denumire": denumire, **dict(zip(COLOANE, valori))})
        elif len(bucati) == 1 and not sume and bal.conturi and linie.startswith(" ") and bucati[0].isupper():
            # Continuarea unei denumiri lungi, pe randul de sub cont
            if not re.search(r"Balanta|Pagina|Debitoare|SAGA", bucati[0]):
                bal.conturi[-1]["denumire"] += " " + bucati[0]

    verifica(bal, totaluri_clase, totaluri_generale)
    return bal


def verifica(bal: Balanta, totaluri_clase: dict, totaluri_generale) -> None:
    """Sumele citite trebuie sa bata cu totalurile tiparite de SAGA; altfel refuzam balanta."""
    if not bal.conturi or not totaluri_clase:
        raise ValueError("Nu am gasit conturi in balanta.")
    for clasa, totaluri in totaluri_clase.items():
        calculat = [round(sum(c[col] for c in bal.conturi if c["cont"][0] == clasa), 2) for col in COLOANE]
        if any(abs(a - b) > 0.02 for a, b in zip(calculat, totaluri)):
            raise ValueError(f"Clasa {clasa}: sumele citite nu bat cu totalul din balanta ({calculat} vs {totaluri}).")
    if totaluri_generale:
        calculat = [round(sum(c[col] for c in bal.conturi), 2) for col in COLOANE]
        if any(abs(a - b) > 0.02 for a, b in zip(calculat, totaluri_generale)):
            raise ValueError("Totalul general citit nu bate cu cel din balanta.")


def indicatori(bal: Balanta) -> dict:
    venituri_an = bal.suma("7", "tr_c")
    venituri_luna = bal.suma("7", "rp_c")
    chelt_an = bal.suma("6", "tr_d")
    chelt_luna = bal.suma("6", "rp_d")
    impozit_profit_an = bal.suma("691", "tr_d") + bal.suma("698", "tr_d")
    profit_brut_an = round(venituri_an - (chelt_an - impozit_profit_an), 2)
    rezultat_reportat = -bal.sold("117")  # pozitiv = profit reportat, negativ = pierdere de acoperit
    profit_net_an = round(profit_brut_an - impozit_profit_an, 2)
    # Capitaluri proprii = conturile 10x, 11x, 12x (fara 16x, imprumuturi). Soldul initial e cel de
    # la sfarsitul anului trecut, adica din ultimele situatii financiare anuale.
    capitaluri = [c for c in bal.conturi if c["cont"][:2] in ("10", "11", "12")]
    cap_inceput = round(sum(c["si_c"] - c["si_d"] for c in capitaluri), 2)
    cap_acum = round(sum(c["sf_c"] - c["sf_d"] for c in capitaluri), 2)
    capital_social = -bal.sold("1012") or -bal.sold("101")
    return {
        "cifra_afaceri_an": round(bal.suma("70", "tr_c"), 2),
        "venituri_an": venituri_an,
        "cheltuieli_an_fara_impozit_profit": round(chelt_an - impozit_profit_an, 2),
        "profit_brut_an": profit_brut_an,
        "impozit_profit_inregistrat_an": round(impozit_profit_an, 2),
        "profit_net_an": profit_net_an,
        "venituri_luna": venituri_luna,
        "cheltuieli_luna": chelt_luna,
        "rezultat_reportat": round(rezultat_reportat, 2),
        "profit_net_an_minus_pierdere_reportata": round(profit_net_an + min(rezultat_reportat, 0), 2),
        "banca_lei": bal.sold("5121"),
        "banca_valuta": bal.sold("5124"),
        "casa": bal.sold("531"),
        "clienti_de_incasat": bal.sold("411"),
        "avansuri_de_la_clienti": -bal.sold("419"),
        "furnizori_de_plata": -(bal.sold("401") + bal.sold("404") + bal.sold("408")),
        "datorie_firma_catre_asociat_455": -bal.sold("455"),
        "debitori_diversi_461": bal.sold("461"),
        "decontari_in_curs_473": bal.sold("473"),
        "impozit_profit_de_plata": -bal.sold("4411"),
        "tva_de_plata": -bal.sold("4423"),
        "tva_de_recuperat": bal.sold("4424"),
        "tva_neexigibila_4428": -bal.sold("4428"),
        "salarii_si_contributii_de_plata": -(bal.sold("421") + bal.sold("431") + bal.sold("436") + bal.sold("444")),
        "alte_taxe_de_plata_446": -bal.sold("446"),
        "capital_social": capital_social,
        "rezerve_legale": -bal.sold("1061"),
        "capitaluri_proprii_sfarsit_an_precedent": cap_inceput,
        "capitaluri_proprii_acum": cap_acum,
        "activ_net_an_precedent_sub_jumatate_capital": 1.0 if cap_inceput < capital_social / 2 else 0.0,
        "impozit_profit_estimat_16_la_suta_din_profit_brut": round(max(profit_brut_an, 0) * 0.16, 2),
        "diferenta_estimata_impozit_profit_neinregistrat": round(
            max(profit_brut_an, 0) * 0.16 - impozit_profit_an, 2),
    }


ETICHETE = {
    "cifra_afaceri_an": "Cifra de afaceri de la inceputul anului (conturi 70x)",
    "venituri_an": "Venituri totale de la inceputul anului",
    "cheltuieli_an_fara_impozit_profit": "Cheltuieli de la inceputul anului (fara impozitul pe profit)",
    "profit_brut_an": "Profit contabil brut de la inceputul anului",
    "impozit_profit_inregistrat_an": "Impozit pe profit inregistrat de la inceputul anului (691)",
    "profit_net_an": "Profit net de la inceputul anului",
    "venituri_luna": "Venituri in luna balantei",
    "cheltuieli_luna": "Cheltuieli in luna balantei",
    "rezultat_reportat": "Rezultat reportat (negativ = pierdere din anii trecuti, de acoperit)",
    "profit_net_an_minus_pierdere_reportata": "Profit net al anului dupa acoperirea pierderii reportate",
    "banca_lei": "Banca lei (5121)",
    "banca_valuta": "Banca valuta (5124)",
    "casa": "Casa (531)",
    "clienti_de_incasat": "Clienti de incasat (411)",
    "avansuri_de_la_clienti": "Avansuri primite de la clienti (419)",
    "furnizori_de_plata": "Furnizori de plata (401, 404, 408)",
    "datorie_firma_catre_asociat_455": "Datoria firmei catre asociat (455; pozitiv = firma ii datoreaza)",
    "debitori_diversi_461": "Debitori diversi (461)",
    "decontari_in_curs_473": "Decontari in curs de clarificare (473)",
    "impozit_profit_de_plata": "Impozit pe profit de plata (4411)",
    "tva_de_plata": "TVA de plata (4423)",
    "tva_de_recuperat": "TVA de recuperat (4424)",
    "tva_neexigibila_4428": "TVA neexigibila (4428, TVA la incasare)",
    "salarii_si_contributii_de_plata": "Salarii si contributii de plata (421, 431, 436, 444)",
    "alte_taxe_de_plata_446": "Alte impozite si taxe de plata (446)",
    "capital_social": "Capital social (1012)",
    "rezerve_legale": "Rezerve legale (1061)",
    "capitaluri_proprii_sfarsit_an_precedent": "Capitaluri proprii (activ net) la sfarsitul anului precedent (conturi 10-12, sold initial)",
    "capitaluri_proprii_acum": "Capitaluri proprii acum, inclusiv profitul anului (conturi 10-12, sold final)",
    "activ_net_an_precedent_sub_jumatate_capital": "Activ net la sfarsitul anului precedent sub jumatate din capitalul social (1 = DA, 0 = nu)",
    "impozit_profit_estimat_16_la_suta_din_profit_brut": "Estimare grosiera: 16% din profitul brut (fara ajustari fiscale)",
    "diferenta_estimata_impozit_profit_neinregistrat": "Estimare grosiera: impozit pe profit inca neinregistrat (16% din profit brut minus 691)",
}


def lei(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ") + " lei"


def rezumat_text(bal: Balanta, ind: dict) -> str:
    """Textul pe care il primeste agentul: indicatorii calculati plus soldurile tuturor conturilor."""
    linii = [
        f"BALANTA {bal.firma} (CUI {bal.cui}), perioada {bal.data_inceput} - {bal.data_sfarsit} (SAGA)",
        "Indicatori calculati din balanta:",
    ]
    linii += [f"- {ETICHETE[k]}: {lei(v)}" for k, v in ind.items()]
    linii.append("Conturi (sold final debitor/creditor; la clasele 6 si 7: total de la inceputul anului):")
    for c in bal.conturi:
        if c["cont"][0] in "67":
            valoare = c["tr_d"] if c["cont"][0] == "6" else c["tr_c"]
            if valoare:
                linii.append(f"  {c['cont']} {c['denumire']}: {lei(valoare)}")
        elif c["sf_d"] or c["sf_c"]:
            linii.append(f"  {c['cont']} {c['denumire']}: D {lei(c['sf_d'])} / C {lei(c['sf_c'])}")
    return "\n".join(linii)


def salveaza(sb, bal: Balanta, ind: dict) -> None:
    """Inlocuieste balanta aceleiasi firme pentru aceeasi perioada (reincarcarea corecteaza)."""
    sb.table("balante").upsert({
        "cui": bal.cui,
        "firma": bal.firma,
        "data_inceput": bal.data_inceput,
        "data_sfarsit": bal.data_sfarsit,
        "conturi": bal.conturi,
        "indicatori": ind,
    }, on_conflict="cui,data_inceput,data_sfarsit").execute()


def ultimele_balante(sb) -> list[dict]:
    """Cea mai recenta balanta pentru fiecare firma."""
    randuri = sb.table("balante").select("cui,firma,data_inceput,data_sfarsit,conturi,indicatori") \
        .order("data_sfarsit", desc=True).limit(50).execute().data
    ultimele: dict[str, dict] = {}
    for r in randuri:
        ultimele.setdefault(r["cui"], r)
    return list(ultimele.values())


def context_financiar(sb) -> str:
    """Blocul cu situatia financiara a firmelor, adaugat la intrebarile agentului."""
    blocuri = []
    for r in ultimele_balante(sb):
        bal = Balanta(r["firma"], r["cui"], r["data_inceput"], r["data_sfarsit"], r["conturi"])
        blocuri.append(rezumat_text(bal, r["indicatori"]))
    return "\n\n".join(blocuri)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    b = citeste_pdf(sys.argv[1])
    i = indicatori(b)
    print(f"{len(b.conturi)} conturi citite, totalurile bat cu balanta.\n")
    print(rezumat_text(b, i))
    if "--salveaza" in sys.argv:
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv()
        salveaza(create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"]), b, i)
        print("\nSalvat in Supabase.")
