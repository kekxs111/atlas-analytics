"""ATLAS Cloud: optionales Login + Depot in Supabase."""
from __future__ import annotations
from typing import Any, Optional
import streamlit as st

def _secrets_get(*keys: str) -> str:
    for k in keys:
        try:
            v = st.secrets.get(k)  # type: ignore[attr-defined]
            if v:
                return str(v).strip()
        except Exception:
            pass
        try:
            if "." in k:
                a, b = k.split(".", 1)
                v = st.secrets.get(a, {}).get(b)  # type: ignore
                if v:
                    return str(v).strip()
        except Exception:
            pass
    return ""


def cloud_configured() -> bool:
    url = _secrets_get("SUPABASE_URL", "supabase.url")
    key = _secrets_get("SUPABASE_ANON_KEY", "supabase.key", "SUPABASE_KEY")
    return bool(url and key)


def get_supabase():
    if not cloud_configured():
        return None
    try:
        from supabase import create_client
    except Exception:
        return None
    url = _secrets_get("SUPABASE_URL", "supabase.url")
    key = _secrets_get("SUPABASE_ANON_KEY", "supabase.key", "SUPABASE_KEY")
    try:
        return create_client(url, key)
    except Exception:
        return None


def finnhub_key() -> str:
    k = _secrets_get("FINNHUB_API_KEY", "finnhub.api_key", "FINNHUB_KEY")
    if k:
        return k
    return str(st.session_state.get("finnhub_key") or "").strip()


def cloud_login(email: str, password: str) -> tuple[bool, str]:
    sb = get_supabase()
    if sb is None:
        return False, "Cloud ist nicht eingerichtet (fehlende Secrets)."
    try:
        res = sb.auth.sign_in_with_password({"email": email, "password": password})
        user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)
        if user is None:
            return False, "Login fehlgeschlagen."
        uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        st.session_state.cloud_user = {"id": uid, "email": email}
        return True, "Angemeldet."
    except Exception as e:
        msg = str(e)
        if "Invalid login" in msg or "invalid" in msg.lower():
            return False, "E-Mail oder Passwort falsch."
        return False, f"Login-Fehler: {msg[:120]}"


def cloud_signup(email: str, password: str) -> tuple[bool, str]:
    sb = get_supabase()
    if sb is None:
        return False, "Cloud ist nicht eingerichtet."
    if len(password) < 6:
        return False, "Passwort mindestens 6 Zeichen."
    try:
        res = sb.auth.sign_up({"email": email, "password": password})
        user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)
        if user is None:
            return False, "Registrierung fehlgeschlagen (E-Mail schon vergeben?)."
        uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        st.session_state.cloud_user = {"id": uid, "email": email}
        return True, "Konto erstellt. Prüfe ggf. deine E-Mails zur Bestätigung."
    except Exception as e:
        return False, f"Registrierung: {str(e)[:120]}"


def cloud_logout() -> None:
    sb = get_supabase()
    try:
        if sb:
            sb.auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("cloud_user", None)


def cloud_user() -> Optional[dict]:
    return st.session_state.get("cloud_user")


def cloud_save_depot(data: dict) -> tuple[bool, str]:
    sb = get_supabase()
    user = cloud_user()
    if sb is None or not user:
        return False, "Nicht angemeldet."
    uid = user.get("id")
    if not uid:
        return False, "Keine User-ID."
    payload = {"user_id": uid, "data": data, "email": user.get("email")}
    try:
        sb.table("depots").upsert(payload, on_conflict="user_id").execute()
        return True, "Depot in der Cloud gespeichert."
    except Exception as e:
        err = str(e)
        if "relation" in err.lower() or "does not exist" in err.lower():
            return False, "Tabelle 'depots' fehlt in Supabase. Siehe Hilfe → Cloud einrichten."
        return False, f"Cloud-Speichern: {err[:140]}"


def cloud_load_depot() -> tuple[bool, Any, str]:
    sb = get_supabase()
    user = cloud_user()
    if sb is None or not user:
        return False, None, "Nicht angemeldet."
    uid = user.get("id")
    try:
        res = sb.table("depots").select("data").eq("user_id", uid).limit(1).execute()
        rows = getattr(res, "data", None) or []
        if not rows:
            return False, None, "Noch kein Cloud-Depot gefunden."
        return True, rows[0].get("data"), "Depot aus der Cloud geladen."
    except Exception as e:
        return False, None, f"Cloud-Laden: {str(e)[:140]}"
