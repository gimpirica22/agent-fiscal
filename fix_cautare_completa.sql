-- Repara cautarea agentului in baza de legi.
-- Indexul are 50 de zone, dar implicit cauta intr-una singura, asa ca rata articole relevante.
-- Supabase nu permite ALTER FUNCTION ... SET ivfflat.probes, asa ca setarea se face in interiorul functiei.
-- Rulare: Supabase Dashboard -> SQL Editor -> Run. Sigur de rulat de mai multe ori.
create or replace function public.match_fiscal_knowledge (
  query_embedding vector(768),
  match_count int default 6
)
returns table (
  id bigint,
  continut text,
  sursa text,
  titlu text,
  articol text,
  similarity float
)
language plpgsql volatile
as $$
begin
  perform set_config('ivfflat.probes', '50', true);
  return query
    select
      f.id,
      f.continut,
      f.sursa,
      f.titlu,
      f.articol,
      1 - (f.embedding <=> query_embedding) as similarity
    from public.fiscal_knowledge f
    order by f.embedding <=> query_embedding
    limit match_count;
end;
$$;
