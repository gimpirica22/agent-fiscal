"""
Bot Telegram — Agent Fiscal Romania

Comenzi disponibile:
    /start   — mesaj de bun venit
    /help    — lista comenzilor
    /intreaba <intrebarea ta> — pune o intrebare fiscala
    Sau trimite direct un mesaj text cu intrebarea.

Configurare in .env:
    TELEGRAM_BOT_TOKEN    — token de la @BotFather
    TELEGRAM_ALLOWED_USERS — lista de user_id Telegram separati prin virgula
                             (lasa gol pentru acces public — nerecomandat)

Rulare:
    python bot_telegram.py
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from supabase import create_client

from agent_fiscal import raspunde
from balanta_saga import citeste_pdf, indicatori, lei, salveaza

load_dotenv()

# Istoricul conversatiei per utilizator: {user_id: [{"role": ..., "text": ...}, ...]}
ISTORII: dict[int, list[dict]] = {}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

ALLOWED_USERS: set[int] = set()
_raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
if _raw:
    ALLOWED_USERS = {int(uid.strip()) for uid in _raw.split(",") if uid.strip()}


def utilizator_permis(user_id: int) -> bool:
    if not ALLOWED_USERS:
        return True  # acces public daca lista e goala
    return user_id in ALLOWED_USERS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not utilizator_permis(update.effective_user.id):
        await update.message.reply_text("Acces neautorizat.")
        return

    await update.message.reply_text(
        "👋 *Agent Fiscal Romania*\n\n"
        "Sunt specializat in Codul Fiscal (Legea 227/2015) si Normele metodologice (H.G. 1/2016).\n\n"
        "Firmele tale:\n"
        "• Step Construct SRL (CUI 35754740)\n"
        "• Total Tehnoconstruct SRL (CUI 40980086)\n\n"
        "Trimite-mi direct intrebarea ta fiscala sau foloseste /intreaba.\n\n"
        "Exemple:\n"
        "• _Ce cheltuieli pot deduce pentru un autoturism?_\n"
        "• _Cum se calculeaza TVA la constructii?_\n"
        "• _Ce impozit platesc pe dividende?_",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not utilizator_permis(update.effective_user.id):
        await update.message.reply_text("Acces neautorizat.")
        return

    await update.message.reply_text(
        "*Comenzi disponibile:*\n\n"
        "/start — mesaj de bun venit\n"
        "/help — aceasta lista\n"
        "/intreaba \\<intrebare\\> — intrebare fiscala\n\n"
        "Sau trimite direct mesajul cu intrebarea ta.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def proceseaza_intrebare(update: Update, context: ContextTypes.DEFAULT_TYPE, intrebare: str) -> None:
    if not utilizator_permis(update.effective_user.id):
        await update.message.reply_text("Acces neautorizat.")
        return

    if len(intrebare.strip()) < 5:
        await update.message.reply_text("Te rog formuleaza o intrebare mai clara.")
        return

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    try:
        user_id = update.effective_user.id
        history = ISTORII.get(user_id, [])
        raspuns, history_nou = raspunde(intrebare, history=history, verbose=False)
        ISTORII[user_id] = history_nou

        # Imparte in bucati de max 4000 caractere la limita de paragraf
        bucati = []
        rest = raspuns
        while rest:
            if len(rest) <= 4000:
                bucati.append(rest)
                break
            taietura = rest.rfind("\n\n", 0, 4000)
            if taietura == -1:
                taietura = rest.rfind("\n", 0, 4000)
            if taietura == -1:
                taietura = 4000
            bucati.append(rest[:taietura])
            rest = rest[taietura:].lstrip()

        for bucata in bucati:
            try:
                await update.message.reply_text(bucata, parse_mode=ParseMode.MARKDOWN)
            except Exception:
                # Fallback: trimitere fara formatare daca Markdown-ul e invalid
                await update.message.reply_text(bucata)

    except Exception as e:
        logger.error("Eroare la procesarea intrebarii: %s", e)
        await update.message.reply_text(
            "A aparut o eroare la procesarea intrebarii. Te rog incearca din nou."
        )


async def cmd_nou(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not utilizator_permis(update.effective_user.id):
        await update.message.reply_text("Acces neautorizat.")
        return
    ISTORII.pop(update.effective_user.id, None)
    await update.message.reply_text("Conversatie resetata. Porneste un subiect nou.")


async def cmd_intreaba(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    intrebare = " ".join(context.args) if context.args else ""
    if not intrebare:
        await update.message.reply_text(
            "Foloseste: /intreaba <intrebarea ta>\n\n"
            "Exemplu: /intreaba Ce cheltuieli pot deduce pentru un autoturism?"
        )
        return
    await proceseaza_intrebare(update, context, intrebare)


async def mesaj_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    intrebare = update.message.text or ""
    await proceseaza_intrebare(update, context, intrebare)


async def document_primit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Balanta de verificare SAGA trimisa ca PDF: o citeste, o verifica si o salveaza."""
    if not utilizator_permis(update.effective_user.id):
        await update.message.reply_text("Acces neautorizat.")
        return
    doc = update.message.document
    if not (doc.file_name or "").lower().endswith(".pdf"):
        await update.message.reply_text("Trimite balanta de verificare exportata din SAGA, in format PDF.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        fisier = await doc.get_file()
        continut = bytes(await fisier.download_as_bytearray())
        bal = citeste_pdf(continut)
        ind = indicatori(bal)
        sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
        salveaza(sb, bal, ind)
    except ValueError as e:
        await update.message.reply_text(f"Nu am putut folosi fisierul: {e}")
        return
    except Exception as e:
        logger.error("Eroare la balanta: %s", e)
        await update.message.reply_text("A aparut o eroare la citirea balantei. Incearca din nou.")
        return

    await update.message.reply_text(
        f"Am salvat balanta {bal.firma}, {bal.data_inceput} - {bal.data_sfarsit} "
        f"({len(bal.conturi)} conturi, totalurile bat).\n\n"
        f"Cifra de afaceri de la inceputul anului: {lei(ind['cifra_afaceri_an'])}\n"
        f"Profit net de la inceputul anului: {lei(ind['profit_net_an'])}\n"
        f"Bani in banca si casa: {lei(ind['banca_lei'] + ind['banca_valuta'] + ind['casa'])}\n"
        f"Clienti de incasat: {lei(ind['clienti_de_incasat'])}\n\n"
        "Iti fac acum analiza..."
    )
    ISTORII.pop(update.effective_user.id, None)
    await proceseaza_intrebare(
        update, context,
        f"Am incarcat balanta {bal.firma} pe perioada {bal.data_inceput} - {bal.data_sfarsit}. "
        "Fa-mi o analiza scurta: cum sta firma, ce ti se pare anormal si de verificat cu contabilul, "
        "ce termene si plati urmeaza si 2-3 optimizari concrete pe cifrele ei.",
    )


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN lipseste din .env")

    print("Bot fiscal pornit. Apasa Ctrl+C pentru a opri.")
    if ALLOWED_USERS:
        print(f"Acces restrictionat la user IDs: {ALLOWED_USERS}")
    else:
        print("ATENTIE: Acces public (seteaza TELEGRAM_ALLOWED_USERS in .env pentru restrictionare)")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("nou", cmd_nou))
    app.add_handler(CommandHandler("intreaba", cmd_intreaba))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mesaj_text))
    app.add_handler(MessageHandler(filters.Document.ALL, document_primit))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
