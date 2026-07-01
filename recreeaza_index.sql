-- ====================================================
-- PAS 1: Ruleaza asta O SINGURA DATA in Supabase SQL Editor
-- Creeaza o functie care poate fi apelata din Python
-- ====================================================
CREATE OR REPLACE FUNCTION recreate_ivfflat_index()
RETURNS text AS $$
BEGIN
  DROP INDEX IF EXISTS fiscal_knowledge_embedding_idx;
  CREATE INDEX fiscal_knowledge_embedding_idx
    ON fiscal_knowledge
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
  RETURN 'Index recreat cu lists=50 (' || (SELECT COUNT(*) FROM fiscal_knowledge)::text || ' randuri)';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ====================================================
-- PAS 2: Dupa ce ai creat functia, Python o apeleaza automat
-- Sau o poti rula manual oricand:
-- ====================================================
-- SELECT recreate_ivfflat_index();
