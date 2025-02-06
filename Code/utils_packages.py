# -*- coding: utf-8 -*-
import importlib
import subprocess
import sys

def install_and_import(package_name, version=None, extra_index_url=None):
    """
    Installiert ein Paket (mit optionaler Version) und importiert es.
    Wenn eine inkompatible Version installiert ist, wird diese entfernt.
    """
    try:
        # Ueberpruefen, ob das Paket importiert werden kann
        module = importlib.import_module(package_name)

        # Pruefen, ob eine spezifische Version erforderlich ist
        if version:
            # Aktuelle Version des Pakets abrufen
            installed_version = subprocess.check_output(
                [sys.executable, "-m", "pip", "show", package_name],
                text=True
            ).splitlines()

            for line in installed_version:
                if line.startswith("Version:"):
                    current_version = line.split(":")[1].strip()
                    if current_version != version:
                        print(f"Inkompatible Version {current_version} gefunden. Entferne {package_name}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
                        raise ImportError  # Erneut installieren
    except ImportError:
        print(f"{package_name} wird installiert...")
        try:
            # Installationsbefehl zusammenstellen
            install_command = [sys.executable, "-m", "pip", "install", package_name]
            if version:
                install_command[-1] = f"{package_name}=={version}"  # Version hinzufuegen
            if extra_index_url:
                install_command.extend(["--extra-index-url", extra_index_url])
            install_command.append("--no-cache-dir")
            subprocess.check_call(install_command, text=True)
        except subprocess.CalledProcessError:
            print(f"Fehler bei der Installation von {package_name}. Bitte pruefen Sie die Paketverfuegbarkeit.")
            return
    finally:
        try:
            # Importiere das Paket nach der Installation
            globals()[package_name] = importlib.import_module(package_name)
        except ModuleNotFoundError:
            print(f"Das Modul {package_name} konnte nach der Installation nicht importiert werden.")