# ATLAS einrichten: Finnhub + Cloud

## 1. Finnhub (stabile Kurse, kostenlos)

1. Gehe zu https://finnhub.io und registriere dich.
2. Kopiere deinen **API Key**.
3. **Option A – in der App:** Sidebar → „Kurse (Finnhub)“ → Key einfügen.
4. **Option B – Streamlit Cloud Secrets:**

```
FINNHUB_API_KEY = "dein-key-hier"
```

Ohne Key funktioniert ATLAS weiter mit Yahoo.

## 2. Cloud-Login (Supabase, kostenlos)

### Projekt
1. https://supabase.com → neues Projekt.
2. Warte bis das Projekt bereit ist.

### Tabelle anlegen (SQL Editor)

```sql
create table if not exists depots (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  data jsonb,
  updated_at timestamptz default now()
);
alter table depots enable row level security;
create policy "own depot" on depots for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
```

### Secrets in Streamlit Cloud

Settings → Secrets:

```
SUPABASE_URL = "https://DEIN-PROJEKT.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
FINNHUB_API_KEY = "dein-finnhub-key"
```

URL und anon key findest du in Supabase unter **Project Settings → API**.

### Nutzen
- App neu starten / redeployen.
- In der Sidebar: **Anmelden / Registrieren**.
- **In Cloud speichern** / **Aus Cloud laden**.

Ohne Cloud-Secrets bleibt nur der lokale Download/Upload.
