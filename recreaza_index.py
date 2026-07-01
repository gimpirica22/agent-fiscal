"""
Asteapta finalizarea ingestion si recreaza indexul IVFFLAT.

Presupune ca functia recreate_ivfflat_index() exista deja in Supabase
(ruleaza recreeaza_index.sql in Supabase SQL Editor o singura data).

Rulare: python recreaza_index.py
"""

import os
import time

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # Verifica cate randuri sunt acum
    rez = sb.table("fiscal_knowledge").select("id", count="exact").limit(1).execute()
    print(f"Randuri curente in fiscal_knowledge: {rez.count}")

    print("Recreez indexul IVFFLAT cu lists=50...")
    start = time.time()

    rezultat = sb.rpc("recreate_ivfflat_index", {}).execute()

    elapsed = time.time() - start
    print(f"Gata in {elapsed:.1f}s: {rezultat.data}")


if __name__ == "__main__":
    main()
