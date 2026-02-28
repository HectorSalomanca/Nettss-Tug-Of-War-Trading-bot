"""
Secret Vault — macOS Keychain integration for Tug-Of-War credentials

Replaces plaintext .env secrets with encrypted-at-rest macOS Keychain storage.
Uses the `security` CLI (built into macOS) — no external dependencies.

Usage:
  1. First time: run `python3 secret_vault.py --store` to migrate .env → Keychain
  2. After that: all modules call get_secret("ALPACA_API_KEY") instead of os.getenv()
  3. Delete .env after confirming Keychain works (keep .env.example for reference)

Security model:
  - Secrets encrypted by macOS Keychain (AES-256, hardware-backed on T2/M1)
  - Only the current user can read them (no root access needed)
  - Survives reboots, survives .env deletion
  - Cannot be accidentally committed to git
  - launchctl processes inherit Keychain access from the user session

Fallback: if Keychain read fails, falls back to os.getenv() (backward compatible)
"""

import os
import re
import subprocess
import sys
import logging
from typing import Optional
from functools import lru_cache

# Keychain service name (groups all TugOfWar secrets together)
KEYCHAIN_SERVICE = "com.tugofwar.trading"

# All secrets the system needs
SECRET_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_ANON_KEY",
    "USER_ID",
]


def _keychain_read(key: str) -> Optional[str]:
    """Read a secret from macOS Keychain. Returns None if not found."""
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-s", KEYCHAIN_SERVICE,
                "-a", key,
                "-w",  # output password only
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _keychain_write(key: str, value: str) -> bool:
    """Write a secret to macOS Keychain. Overwrites if exists."""
    try:
        # Delete existing entry first (security add fails if duplicate)
        subprocess.run(
            ["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", key],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "security", "add-generic-password",
                "-s", KEYCHAIN_SERVICE,
                "-a", key,
                "-w", value,
                "-U",  # update if exists
            ],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _keychain_delete(key: str) -> bool:
    """Delete a secret from macOS Keychain."""
    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", key],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── Secret leak prevention ──────────────────────────────────────────────────
# Collect loaded secrets so we can scrub them from any output.
_loaded_secrets: list[str] = []


def scrub(text: str) -> str:
    """Replace any loaded secret value in *text* with [REDACTED]."""
    for s in _loaded_secrets:
        if s and len(s) > 4:
            text = text.replace(s, "[REDACTED]")
    return text


class _SecretScrubFilter(logging.Filter):
    """Logging filter that strips secrets from every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = scrub(record.msg)
        if record.args:
            record.args = tuple(
                scrub(a) if isinstance(a, str) else a for a in record.args
            )
        return True


def install_scrub_filter():
    """Attach the secret-scrub filter to the root logger (idempotent)."""
    root = logging.getLogger()
    if not any(isinstance(f, _SecretScrubFilter) for f in root.filters):
        root.addFilter(_SecretScrubFilter())


# Also monkey-patch builtins.print so even raw print() calls are scrubbed
import builtins
_original_print = builtins.print

def _safe_print(*args, **kwargs):
    safe_args = tuple(scrub(str(a)) if isinstance(a, str) else a for a in args)
    _original_print(*safe_args, **kwargs)

builtins.print = _safe_print


@lru_cache(maxsize=32)
def get_secret(key: str) -> Optional[str]:
    """
    Get a secret with Keychain-first, .env-fallback strategy.

    Priority:
      1. macOS Keychain (encrypted at rest)
      2. Environment variable (from .env or shell)
      3. None
    """
    # Try Keychain first
    value = _keychain_read(key)
    if value:
        _loaded_secrets.append(value)
        return value

    # Fallback to environment variable
    value = os.getenv(key)
    if value:
        _loaded_secrets.append(value)
        return value

    return None


# Auto-install the scrub filter on import
install_scrub_filter()


def store_env_to_keychain(env_path: str = ".env") -> dict:
    """
    Migrate all secrets from .env file to macOS Keychain.
    Returns dict of {key: success_bool}.
    """
    from dotenv import dotenv_values

    if not os.path.exists(env_path):
        print(f"[VAULT] .env file not found at {env_path}")
        return {}

    values = dotenv_values(env_path)
    results = {}

    for key in SECRET_KEYS:
        value = values.get(key)
        if not value:
            print(f"[VAULT] {key}: not found in .env — skipping")
            results[key] = False
            continue

        success = _keychain_write(key, value)
        if success:
            print(f"[VAULT] {key}: ✓ stored in Keychain")
        else:
            print(f"[VAULT] {key}: ✗ Keychain write failed")
        results[key] = success

    return results


def verify_keychain() -> dict:
    """Verify all secrets are readable from Keychain."""
    results = {}
    for key in SECRET_KEYS:
        value = _keychain_read(key)
        if value:
            # Mask the value for display
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            print(f"[VAULT] {key}: ✓ ({masked})")
            results[key] = True
        else:
            # Check .env fallback
            env_val = os.getenv(key)
            if env_val:
                print(f"[VAULT] {key}: ⚠ not in Keychain, using .env fallback")
                results[key] = "fallback"
            else:
                print(f"[VAULT] {key}: ✗ not found anywhere")
                results[key] = False
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tug-Of-War Secret Vault")
    parser.add_argument("--store", action="store_true", help="Migrate .env secrets to macOS Keychain")
    parser.add_argument("--verify", action="store_true", help="Verify all secrets are accessible")
    parser.add_argument("--get", type=str, help="Get a specific secret (masked)")
    args = parser.parse_args()

    if args.store:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(base, ".env")
        print(f"[VAULT] Migrating secrets from {env_path} to macOS Keychain...")
        store_env_to_keychain(env_path)
        print("\n[VAULT] Verifying migration:")
        verify_keychain()
        print("\n[VAULT] Done. You can now delete .env (keep .env.example for reference).")
        print("[VAULT] All modules will auto-fallback to .env if Keychain read fails.")
    elif args.verify:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        verify_keychain()
    elif args.get:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
        val = get_secret(args.get)
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
            print(f"{args.get} = {masked}")
        else:
            print(f"{args.get} = NOT FOUND")
    else:
        parser.print_help()
