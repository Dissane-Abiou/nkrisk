"""
Rendu HTML d'un rapport de validation — la couche de présentation.

POURQUOI C'EST UN MODULE SÉPARÉ
-------------------------------
`report.py` contenait, avant la séance 15, la règle de conjonction, la
détection de conflits, le repliement par régime… et deux cents lignes
de CSS. Mille vingt lignes au total, où la logique métier et le format
de sortie se touchaient.

Rien ne cassait. C'est précisément le problème d'un défaut de couches :
il ne casse pas, il rend tout plus cher. Changer une couleur obligeait
à rouvrir le fichier qui décide si un modèle est rejeté, et le module
qui porte les règles de validation ne pouvait pas être lu sans faire
défiler une feuille de style.

La séparation donne trois choses concrètes :

  — le domaine ne dépend plus d'un format de sortie, donc un second
    rendu (Markdown, JSON, PDF) s'ajoute sans y toucher ;
  — les tests de règles ne chargent plus le CSS ;
  — un relecteur qui veut vérifier la règle de conjonction lit trois
    cents lignes au lieu de mille.

`ValidationReport.to_html()` reste, et délègue ici. C'est une commodité
assumée : le domaine n'a pas besoin de connaître le HTML, mais
l'utilisateur n'a pas envie d'importer deux modules pour obtenir un
fichier.

CONTRAINTE DE SORTIE, ET ELLE N'EST PAS COSMÉTIQUE
---------------------------------------------------
Le fichier produit est AUTONOME : aucune feuille de style externe,
aucun script, aucune police chargée depuis ailleurs. Un rapport de
validation doit pouvoir être joint à un courriel, archivé, et rouvert
dans dix ans. Tout ce qui est chargé depuis ailleurs aura disparu.

Tout ce qui vient des données — noms d'actifs, nom d'univers, détails
d'anomalies — passe par `html.escape`. C'est la discipline de la
séance 10 appliquée à une SORTIE plutôt qu'à une entrée, et
`test_le_html_echappe_ce_qui_vient_des_donnees` la verrouille.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import pandas as pd

from .validation import REJETE, SURVEILLANCE, VALIDE

if TYPE_CHECKING:  # pragma: no cover - uniquement pour les annotations
    from .report import ValidationReport

#: Importés depuis le domaine pour que les deux modules ne puissent pas
#: diverger sur les noms de familles.
from .report import PERIMETRE, SEVERITE

_CSS = """
:root{--bg:#fbfaf8;--fg:#1a1a1a;--muted:#6b6b6b;--line:#e2ded8;--card:#fff;
--ok:#1f7a4d;--warn:#9a6b00;--bad:#a8301a;--accent:#2d4a7c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#14161a;--fg:#e8e6e3;--muted:#9a9894;--line:#2a2e35;--card:#1b1e24;
--ok:#4bbd84;--warn:#d9a441;--bad:#e2705c;--accent:#7aa2d8}}
:root[data-theme="dark"]{--bg:#14161a;--fg:#e8e6e3;--muted:#9a9894;
--line:#2a2e35;--card:#1b1e24;--ok:#4bbd84;--warn:#d9a441;--bad:#e2705c;
--accent:#7aa2d8}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:48px 16px;
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:1.05rem;margin:40px 0 12px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted);font-weight:600}
.sub{color:var(--muted);font-size:.9rem;margin-bottom:32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:12px 0}
.conflict{border-left:3px solid var(--warn)}
.conflict.perimetre{border-left-color:var(--bad)}
.fam{font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
color:var(--warn);font-weight:700}
.conflict.perimetre .fam{color:var(--bad)}
.subj{font-weight:600;margin:4px 0}
.say{color:var(--muted);font-size:.88rem;margin:6px 0}
.pre{white-space:pre-line}
.interp{margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:.88rem;margin-top:8px}
th{text-align:left;font-weight:600;color:var(--muted);font-size:.75rem;
text-transform:uppercase;letter-spacing:.06em;padding:8px 10px;
border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.v{font-weight:600;font-size:.82rem;white-space:nowrap}
.v-ok{color:var(--ok)}.v-warn{color:var(--warn)}.v-bad{color:var(--bad)}
.v-na{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem}
.caveat{color:var(--bad);font-size:.88rem;margin:6px 0}
.note{color:var(--muted);font-size:.84rem;border-top:1px solid var(--line);
margin-top:40px;padding-top:16px}
.kv{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
.kv span{font-size:.82rem;color:var(--muted)}
@media (max-width:640px){body{padding:28px 16px}
table{display:block;overflow-x:auto}}
"""


def _classe(verdict: str) -> str:
    if verdict == VALIDE:
        return "v-ok"
    if verdict == SURVEILLANCE:
        return "v-warn"
    if verdict == REJETE:
        return "v-bad"
    return "v-na"


def render_html(r: ValidationReport) -> str:
    e = html.escape
    out: list[str] = []
    out.append("<!doctype html><html lang='fr'><head><meta charset='utf-8'>")
    out.append("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    out.append(f"<title>Validation — {e(r.universe)}</title>")
    out.append(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    out.append(f"<h1>Rapport de validation — {e(r.universe)}</h1>")
    out.append(
        f"<div class='sub'>{r.n_assets} actifs · {r.n_obs} observations · "
        f"{e(r.window[0])} → {e(r.window[1])} · "
        f"queue à {r.alpha:.1%} · produit le {e(r.generated)} par "
        f"<span class='mono'>nkrisk</span></div>"
    )

    if r.data_caveats:
        out.append("<h2>Données — ce qui conditionne tout le reste</h2>")
        out.append("<div class='card'>")
        for avertissement in r.data_caveats:
            out.append(f"<div class='caveat'>⚠ {e(avertissement)}</div>")
        out.append("</div>")

    perimetre = [c for c in r.conflicts if c.family == PERIMETRE]
    autres = [c for c in r.conflicts if c.family != PERIMETRE]
    if r.conflicts:
        out.append(f"<h2>Conflits entre instruments ({len(r.conflicts)})</h2>")
        for conflit in perimetre + autres:
            cls = (
                "conflict perimetre" if conflit.family == PERIMETRE else "conflict"
            )
            out.append(f"<div class='card {cls}'>")
            out.append(f"<div class='fam'>{e(conflit.family)}</div>")
            out.append(f"<div class='subj'>{e(conflit.subject)}</div>")
            for i, v in zip(conflit.instruments, conflit.verdicts, strict=False):
                out.append(f"<div class='say'>{e(i)} → {e(v)}</div>")
            out.append(f"<div class='interp'>{e(conflit.interpretation)}</div>")
            out.append("</div>")

    out.append("<h2>Verdicts</h2><div class='card'>")
    table = r.table()
    if table.empty:
        out.append("<div class='say'>aucun modèle jugé</div>")
    else:
        colonnes = list(table.columns)
        out.append("<table><thead><tr>")
        for c in colonnes:
            out.append(f"<th>{e(str(c))}</th>")
        out.append("</tr></thead><tbody>")
        for _, ligne in table.iterrows():
            out.append("<tr>")
            for c in colonnes:
                valeur = ligne[c]
                texte = "—" if pd.isna(valeur) else str(valeur)
                if texte in SEVERITE:
                    out.append(
                        f"<td><span class='v {_classe(texte)}'>{e(texte)}</span></td>"
                    )
                else:
                    out.append(f"<td>{e(texte)}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
    out.append("</div>")

    out.append("<h2>Détail des constats</h2><div class='card'>")
    for s in r.subjects:
        out.append(f"<div class='subj'>{e(s)}</div><div class='kv'>")
        for f in r.findings_for(s):
            out.append(
                f"<span><b class='{_classe(f.verdict)}'>{e(f.instrument)}</b> "
                f"{e(f.detail)}</span>"
            )
        out.append("</div>")
    out.append("</div>")

    out.append("<h2>Contrôles transversaux</h2><div class='card'>")
    out.append(f"<div class='say'><b>Causalité</b> — {e(r.causality)}</div>")
    out.append(
        f"<div class='say pre'><b>Concentration</b>\n{e(r.concentration)}</div>"
    )
    out.append("</div>")

    out.append(
        "<div class='note'><b>Règle de conjonction.</b> Le verdict d'un modèle "
        "est le PIRE des verdicts rendus sur lui. Une absence de preuve ne "
        "compense pas une preuve : trois validations n'annulent pas un rejet, "
        "elles disent que le défaut est hors du champ de trois instruments. La "
        "colonne « contraint par » nomme l'instrument qui décide — c'est lui "
        "qui dit quoi réparer."
        f"<br><br><b>Seuil de dispersion</b> : {r.dispersion_max:.2f} × le hasard. "
        "C'est une convention de gouvernance, pas une propriété des données."
        "<br><br><b>Ce que ce rapport ne sait pas faire.</b> Il ne cherche que "
        "les conflits qu'on lui a appris à reconnaître. Deux instruments "
        "peuvent diverger d'une façon qu'il ne voit pas — exactement comme un "
        "test ne voit que ce que sa statistique regarde.</div>"
    )

    out.append("</div></body></html>")
    return "\n".join(out)
