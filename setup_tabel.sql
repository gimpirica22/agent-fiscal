create extension if not exists vector;

create table if not exists fiscal_knowledge (
  id bigserial primary key,
  continut text not null,
  embedding vector(768),
  sursa text,
  titlu text,
  articol text,
  chunk_index integer,
  created_at timestamp default now()
);

create index if not exists fiscal_knowledge_embedding_idx on fiscal_knowledge
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- Functie pentru cautare prin similaritate (folosita ulterior de agent)
create or replace function match_fiscal_knowledge (
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
language sql stable
as $$
  select
    id,
    continut,
    sursa,
    titlu,
    articol,
    1 - (embedding <=> query_embedding) as similarity
  from fiscal_knowledge
  order by embedding <=> query_embedding
  limit match_count;
$$;
