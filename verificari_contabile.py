"""
Agentul contabil: verifica balantele SAGA cu reguli fixe si scrie constatarile.

Verificarile sunt calculate aici, in Python, nu de modelul AI: fiecare constatare are contul,
suma si motivul. Agentul fiscal primeste apoi constatarile si explica ce e de facut.

Niveluri:
    EROARE   - sigur gresit in contabilitate (sold imposibil, balanta neechilibrata)
    ATENTIE  - de verificat cu contabilul (poate fi corect, dar de obicei nu e)

Rulare (test pe balantele din Supabase):
    python verificari_contabile.py
"""

import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta

from balanta_saga import Balanta, lei

PRAG = 1.0  # lei; diferentele mai mici sunt rotunjiri


@dataclass
class Constatare:
    nivel: str  # EROARE / ATENTIE
    cont: str
    mesaj: str


def sold_final(c: dict) -> float:
    """Pozitiv = debitor, negativ = creditor."""
    return round(c["sf_d"] - c["sf_c"], 2)


def sold_inceput_luna(c: dict) -> float:
    """Soldul la inceputul lunii balantei: sold initial an + rulajele lunilor anterioare."""
    return round((c["si_d"] - c["si_c"]) + (c["tr_d"] - c["rp_d"]) - (c["tr_c"] - c["rp_c"]), 2)


def incepe(cont: str, prefixe) -> bool:
    return any(cont.startswith(p) for p in prefixe)


# Conturi care nu pot avea sold creditor (cont, nivel, explicatie)
NUMAI_DEBITOARE = [
    (("5311", "5314"), "EROARE", "casa nu poate avea sold creditor (s-a platit mai mult numerar decat exista)"),
    (("5121", "5124"), "EROARE", "contul de banca nu poate fi creditor; descoperitul de cont se inregistreaza in 5191"),
    (("4111", "4118"), "ATENTIE", "clienti cu sold creditor: avansurile primite se inregistreaza in 419"),
    (("409",), "ATENTIE", "furnizori-debitori cu sold creditor"),
    (("4424",), "EROARE", "TVA de recuperat nu poate fi creditoare"),
    (("4426",), "EROARE", "TVA deductibila nu poate fi creditoare"),
    (("461",), "ATENTIE", "debitori diversi cu sold creditor"),
    (("542",), "ATENTIE", "avansuri de trezorerie cu sold creditor"),
    (("20", "21", "22", "23"), "EROARE", "imobilizarile nu pot avea sold creditor"),
]
# Conturi care nu pot avea sold debitor
NUMAI_CREDITOARE = [
    (("401", "404", "408"), "ATENTIE", "furnizori cu sold debitor: avansurile date se inregistreaza in 409"),
    (("419",), "ATENTIE", "clienti-creditori cu sold debitor"),
    (("421",), "ATENTIE", "salarii cu sold debitor (s-a platit mai mult decat s-a calculat)"),
    (("431", "436", "437", "444"), "ATENTIE", "contributii/impozit pe salarii cu sold debitor (platit in plus)"),
    (("4411",), "ATENTIE", "impozit pe profit platit in plus (sold debitor)"),
    (("4423",), "EROARE", "TVA de plata nu poate fi debitoare"),
    (("4427",), "EROARE", "TVA colectata nu poate fi debitoare"),
    (("4428",), "ATENTIE", "TVA neexigibila cu sold debitor: verifica facturile cu TVA la incasare"),
    (("462",), "ATENTIE", "creditori diversi cu sold debitor"),
    (("455",), "ATENTIE", "asociatul datoreaza bani firmei (455 debitor): sumele luate de asociat "
                          "fara document pot fi tratate fiscal ca venit al lui"),
    (("28", "29"), "EROARE", "amortizarile si ajustarile nu pot avea sold debitor"),
    (("101",), "EROARE", "capitalul social nu poate fi debitor"),
]
# Stocuri: numai debitoare, cu exceptia conturilor de diferente de pret
STOCURI_BIFUNCTIONALE = ("308", "348", "358", "368", "378", "388", "39")

# Datorii la buget/salariati verificate la plata
DATORII_DE_PLATA = {
    "4411": "impozitul pe profit",
    "4423": "TVA de plata",
    "4315": "CAS",
    "4316": "CASS",
    "436": "contributia asiguratorie de munca",
    "444": "impozitul pe salarii",
    "446": "alte impozite si taxe",
    "421": "salariile",
}

# Amortizare (281x) fata de imobilizarea amortizata
AMORTIZARI = {"2801": "201", "2803": "203", "2805": "205", "2807": "207", "2808": "208",
              "2812": "212", "2813": "213", "2814": "214"}


def luna(bal: Balanta) -> int:
    return int(bal.data_sfarsit[5:7])


def verifica_balanta(bal: Balanta, anterioara: Balanta | None = None) -> list[Constatare]:
    rez: list[Constatare] = []
    adauga = lambda nivel, cont, mesaj: rez.append(Constatare(nivel, cont, mesaj))

    # 1. Balanta echilibrata: total debit = total credit pe fiecare coloana
    for d, c, nume in (("si_d", "si_c", "soldurile initiale"), ("rp_d", "rp_c", "rulajele lunii"),
                       ("tr_d", "tr_c", "rulajele cumulate"), ("sf_d", "sf_c", "soldurile finale")):
        dif = round(sum(x[d] for x in bal.conturi) - sum(x[c] for x in bal.conturi), 2)
        if abs(dif) > 0.02:
            adauga("EROARE", "-", f"Balanta nu e echilibrata la {nume}: debit - credit = {lei(dif)}.")

    for c in bal.conturi:
        cont, sold = c["cont"], sold_final(c)
        den = f"{cont} {c['denumire']}"

        # 2. Solduri negative (storno mai mare decat soldul)
        if c["sf_d"] < -0.005 or c["sf_c"] < -0.005:
            adauga("EROARE", cont, f"{den}: sold final negativ ({lei(c['sf_d'] or c['sf_c'])}). "
                                   "De obicei e o stornare gresita sau o inregistrare pe contul gresit.")

        # 3. Sold pe partea gresita
        if sold < -PRAG:
            for prefixe, nivel, expl in NUMAI_DEBITOARE:
                if incepe(cont, prefixe):
                    adauga(nivel, cont, f"{den}: sold creditor {lei(-sold)} - {expl}.")
                    break
            else:
                if cont[0] == "3" and not incepe(cont, STOCURI_BIFUNCTIONALE):
                    adauga("EROARE", cont, f"{den}: stoc negativ {lei(-sold)} - s-a dat in consum "
                                           "mai mult decat s-a receptionat.")
        elif sold > PRAG:
            for prefixe, nivel, expl in NUMAI_CREDITOARE:
                if incepe(cont, prefixe):
                    adauga(nivel, cont, f"{den}: sold debitor {lei(sold)} - {expl}.")
                    break

        # 4. Conturile de venituri si cheltuieli inchise la sfarsitul lunii
        if cont[0] in "67" and abs(sold) > PRAG:
            adauga("ATENTIE", cont, f"{den}: are sold {lei(abs(sold))} - luna nu a fost inchisa "
                                    "(conturile de venituri si cheltuieli se inchid prin 121).")

    # 5. Conturi de asteptare / neclarificate
    for prefix, text in (("473", "decontari in curs de clarificare: trebuie lamurite si trecute pe contul "
                                 "corect, cel tarziu la inchiderea anului"),
                         ("461", "debitori diversi: verifica cine datoreaza si din ce document; sumele "
                                 "fara acte pot fi recalificate la control"),
                         ("542", "avansuri de trezorerie nedecontate: trebuie justificate cu documente")):
        for c in bal.cont(prefix):
            sold = sold_final(c)
            if sold > PRAG:
                vechi = c["si_d"] - c["si_c"]
                extra = f" (din care {lei(vechi)} existau deja la inceputul anului)" if vechi > PRAG else ""
                adauga("ATENTIE", c["cont"], f"{c['cont']} sold {lei(sold)}{extra} - {text}.")

    # 6. Casa
    casa = bal.sold("531")
    if casa > 10_000:
        adauga("ATENTIE", "531",
               f"Sold de casa {lei(casa)}. Verifica daca banii sunt fizic in casierie: soldul din registrul "
               "de casa trebuie sa fie numerarul real. Un sold scriptic mare e unul dintre primele lucruri "
               "verificate la control.")

    # 7. Datorii la buget si salarii neplatite in luna
    for c in bal.conturi:
        if c["cont"] not in DATORII_DE_PLATA:
            continue
        datorat = -sold_inceput_luna(c)
        platit = c["rp_d"]
        if datorat > PRAG and platit < datorat - PRAG:
            ramas = datorat - platit
            if c["cont"] == "421":
                mesaj = (f"421: salarii de {lei(datorat)} datorate la inceputul lunii, platite in luna "
                         f"{lei(platit)}; raman neplatite {lei(ramas)}.")
            else:
                mesaj = (f"{c['cont']} ({DATORII_DE_PLATA[c['cont']]}): {lei(datorat)} de plata la inceputul "
                         f"lunii, platit in luna {lei(platit)}; nu apare plata a {lei(ramas)}. Daca scadenta "
                         "(de regula pe 25) a trecut si plata nu s-a facut, curg dobanzi si penalitati.")
            adauga("ATENTIE", c["cont"], mesaj)

    # 8. TVA inchisa la sfarsit de trimestru
    if luna(bal) in (3, 6, 9, 12):
        for prefix in ("4426", "4427"):
            sold = bal.sold(prefix)
            if abs(sold) > PRAG:
                adauga("ATENTIE", prefix, f"{prefix} are sold {lei(abs(sold))} la sfarsit de trimestru: "
                                          "TVA nu a fost regularizata in 4423/4424 (inchiderea de TVA).")

    # 9. Amortizare
    for cont_am, cont_imob in AMORTIZARI.items():
        amortizat, valoare = -bal.sold(cont_am), bal.sold(cont_imob)
        if amortizat > valoare + PRAG:
            adauga("EROARE", cont_am, f"Amortizarea {cont_am} ({lei(amortizat)}) e mai mare decat valoarea "
                                      f"imobilizarilor {cont_imob} ({lei(valoare)}).")
    neamortizat_la_inceput = sum(
        (c["si_d"] - c["si_c"]) for c in bal.conturi if incepe(c["cont"], ("201", "203", "205", "207", "208",
                                                                          "212", "213", "214"))
    ) - sum((c["si_c"] - c["si_d"]) for c in bal.conturi if c["cont"].startswith("28"))
    if neamortizat_la_inceput > PRAG and luna(bal) >= 2 and bal.suma("6811", "tr_d") < PRAG:
        adauga("ATENTIE", "6811", f"Imobilizari neamortizate de {lei(neamortizat_la_inceput)} la inceputul anului, "
                                  "dar nu s-a inregistrat amortizare in acest an (6811).")

    # 10. Impozitul pe profit trimestrial
    venituri, cheltuieli = bal.suma("7", "tr_c"), bal.suma("6", "tr_d")
    impozit = bal.suma("691", "tr_d") + bal.suma("698", "tr_d")
    profit_brut = venituri - (cheltuieli - impozit)
    if luna(bal) >= 3 and profit_brut > 1000 and impozit < PRAG:
        adauga("ATENTIE", "691", f"Profit brut {lei(profit_brut)} de la inceputul anului, dar nu e inregistrat "
                                 "impozit pe profit (691). Impozitul se calculeaza si se declara trimestrial.")

    # 11. Contul 121 trebuie sa bata cu veniturile minus cheltuielile
    if not any(c["cont"][0] in "67" and abs(sold_final(c)) > PRAG for c in bal.conturi):
        rezultat_121 = -bal.sold("121")
        rezultat_an = round(venituri - cheltuieli, 2)
        rezultat_an_trecut = -sum(c["si_d"] - c["si_c"] for c in bal.cont("121"))
        if abs(rezultat_121 - rezultat_an) > PRAG:
            if abs(rezultat_121 - rezultat_an - rezultat_an_trecut) <= PRAG and luna(bal) >= 6:
                adauga("ATENTIE", "121", f"Rezultatul anului trecut ({lei(rezultat_an_trecut)}) e inca in 121: "
                                         "nu a fost repartizat sau trecut in 117 dupa aprobarea situatiilor anuale.")
            elif abs(rezultat_121 - rezultat_an - rezultat_an_trecut) > PRAG:
                adauga("EROARE", "121", f"Contul 121 ({lei(rezultat_121)}) nu bate cu veniturile minus "
                                        f"cheltuielile anului ({lei(rezultat_an)}).")

    # 12. Luna anterioara modificata dupa ce a fost trimisa balanta
    if anterioara and anterioara.data_sfarsit[:4] == bal.data_sfarsit[:4]:
        acum = {c["cont"]: sold_inceput_luna(c) for c in bal.conturi}
        inainte = {c["cont"]: sold_final(c) for c in anterioara.conturi}
        modificate = [
            f"{cont} ({lei(inainte.get(cont, 0))} -> {lei(acum.get(cont, 0))})"
            for cont in sorted(set(acum) | set(inainte))
            if abs(acum.get(cont, 0) - inainte.get(cont, 0)) > 0.005
        ]
        if modificate:
            adauga("ATENTIE", "-", f"Luna {anterioara.data_sfarsit[:7]} a fost modificata dupa balanta trimisa "
                                   f"(sold final atunci -> sold de inceput acum): {', '.join(modificate[:8])}"
                                   + (f" si inca {len(modificate) - 8}" if len(modificate) > 8 else "")
                                   + ". Daca s-a corectat o declaratie depusa, poate trebuie si declaratie rectificativa.")

    ordine = {"EROARE": 0, "ATENTIE": 1}
    return sorted(rez, key=lambda x: ordine[x.nivel])


def verifica_grup(balante: list[Balanta]) -> list[Constatare]:
    """Verificari intre firmele grupului, pe balante din aceeasi luna."""
    rez = []
    pe_luna: dict[str, list[Balanta]] = {}
    for b in balante:
        pe_luna.setdefault(b.data_sfarsit, []).append(b)
    for data_sf, lista in pe_luna.items():
        cu_451 = [b for b in lista if b.cont("451")]
        if len(cu_451) >= 2:
            total = round(sum(b.sold("451") for b in cu_451), 2)
            if abs(total) > PRAG:
                detalii = ", ".join(f"{b.firma}: {lei(b.sold('451'))}" for b in cu_451)
                rez.append(Constatare("ATENTIE", "451",
                                      f"La {data_sf}, decontarile intre firmele grupului (451) nu se compenseaza "
                                      f"({detalii}; diferenta {lei(total)}). Ce datoreaza o firma trebuie sa apara "
                                      "ca creanta la cealalta."))
    return rez


def din_rand(r: dict) -> Balanta:
    return Balanta(r["firma"], r["cui"], r["data_inceput"], r["data_sfarsit"], r["conturi"])


def balanta_anterioara(sb, bal: Balanta) -> Balanta | None:
    ziua_dinainte = (date.fromisoformat(bal.data_inceput) - timedelta(days=1)).isoformat()
    r = sb.table("balante").select("cui,firma,data_inceput,data_sfarsit,conturi") \
        .eq("cui", bal.cui).eq("data_sfarsit", ziua_dinainte).limit(1).execute().data
    return din_rand(r[0]) if r else None


def text_constatari(bal: Balanta, constatari: list[Constatare]) -> str:
    titlu = f"VERIFICARE CONTABILA {bal.firma}, {bal.data_inceput} - {bal.data_sfarsit}"
    if not constatari:
        return f"{titlu}: nicio problema gasita de verificarile automate."
    return titlu + ":\n" + "\n".join(f"- [{c.nivel}] {c.mesaj}" for c in constatari)


def raport_grup(sb) -> str:
    """Raportul agentului contabil pe ultima balanta a fiecarei firme, plus verificarile de grup."""
    from balanta_saga import ultimele_balante
    balante = [din_rand(r) for r in ultimele_balante(sb)]
    if not balante:
        return "Nicio balanta incarcata."
    blocuri = [text_constatari(b, verifica_balanta(b, balanta_anterioara(sb, b))) for b in balante]
    grup = verifica_grup(balante)
    if grup:
        blocuri.append("VERIFICARI INTRE FIRMELE GRUPULUI:\n" + "\n".join(f"- [{c.nivel}] {c.mesaj}" for c in grup))
    return "\n\n".join(blocuri)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    print(raport_grup(create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])))
