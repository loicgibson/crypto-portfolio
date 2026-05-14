import getpass
import sys

import keyring

from ..config import KEYRING_SERVICE
from ..display import console


def cmd_setup_keys(_args) -> None:
    console.print("[bold]Configuration des clés API Binance[/]")
    console.print("Les clés seront stockées dans le Windows Credential Manager (chiffrées par l'OS).\n")
    api_key    = getpass.getpass("API Key     : ")
    api_secret = getpass.getpass("API Secret  : ")
    if not api_key or not api_secret:
        console.print("[red]Annulé — les clés ne peuvent pas être vides.[/]")
        sys.exit(1)
    keyring.set_password(KEYRING_SERVICE, "api_key", api_key)
    keyring.set_password(KEYRING_SERVICE, "api_secret", api_secret)
    console.print("[green]Clés Binance enregistrées avec succès dans le Credential Manager.[/]")


def cmd_setup_anthropic(_args) -> None:
    console.print("[bold]Configuration de la clé API Anthropic[/]")
    console.print("La clé sera stockée dans le Windows Credential Manager (chiffrée par l'OS).\n")
    api_key = getpass.getpass("Anthropic API Key : ")
    if not api_key:
        console.print("[red]Annulé — la clé ne peut pas être vide.[/]")
        sys.exit(1)
    keyring.set_password(KEYRING_SERVICE, "anthropic_api_key", api_key)
    console.print("[green]Clé Anthropic enregistrée avec succès dans le Credential Manager.[/]")


def cmd_setup_grok(_args) -> None:
    console.print("[bold]Configuration de la clé API Grok (xAI)[/]")
    console.print("Crée un compte sur [link]https://console.x.ai[/link] pour obtenir ta clé.")
    console.print("La clé sera stockée dans le Windows Credential Manager (chiffrée par l'OS).\n")
    api_key = getpass.getpass("xAI API Key : ")
    if not api_key:
        console.print("[red]Annulé — la clé ne peut pas être vide.[/]")
        sys.exit(1)
    keyring.set_password(KEYRING_SERVICE, "grok_api_key", api_key)
    console.print("[green]Clé Grok enregistrée avec succès dans le Credential Manager.[/]")
    console.print("[dim]Le sentiment X sera automatiquement activé lors des prochains cycles.[/]")


def register(sub):
    sub.add_parser("setup-keys",      help="Configurer les clés API Binance dans le Credential Manager")
    sub.add_parser("setup-anthropic", help="Configurer la clé API Anthropic dans le Credential Manager")
    sub.add_parser("setup-grok",      help="Configurer la clé API Grok (xAI) pour le sentiment X")
    return {
        "setup-keys":      cmd_setup_keys,
        "setup-anthropic": cmd_setup_anthropic,
        "setup-grok":      cmd_setup_grok,
    }
