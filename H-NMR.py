# app.py
import re
import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

st.set_page_config(page_title="NMR Trainer (Drag & Drop Molecule)", layout="wide")

st.title("🧪 NMR Trainer: Molekül zeichnen → Bild + ¹H-NMR Übungstabelle")

st.markdown(
    """
Ich zeichne mein Molekül im Editor, klicke **Apply**, und bekomme:
- ein **Molekülbild**
- eine **Tabelle** mit: **H-Gruppen**, **Shift (ppm)**, **Splitting**, **Intensity/Integral**
"""
)

# -----------------------
# 1) Ketcher editor (Drag&Drop) -> SMILES
# -----------------------
st.markdown("### Molekül-Editor")
st.caption("⬇️ Zeichne dein Molekül im Editor (Drag & Drop). Klicke dann **Apply** oben rechts im Editor.")

smiles = st_ketcher(height=500)

# Zeige aktuelle SMILES an
if smiles:
    st.success(f"**SMILES:** `{smiles}`")

if not smiles or not isinstance(smiles, str) or len(smiles.strip()) == 0:
    st.info("Zeichne ein Molekül und klicke **Apply** (oben rechts im Editor).")
    st.stop()

# -----------------------
# 2) RDKit helpers
# -----------------------
def mol_from_smiles(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)  # explicit H for counting
    return mol

def multiplicity_name(n: int) -> str:
    return {
        0: "singlet",
        1: "doublet",
        2: "triplet",
        3: "quartet",
        4: "quintet",
        5: "sextet",
        6: "septet",
    }.get(n, f"{n+1}-plet")

def is_carbonyl_carbon(atom):
    if atom.GetSymbol() != "C":
        return False
    for b in atom.GetBonds():
        if b.GetBondType() == BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False

def is_hetero(atom):
    return atom.GetSymbol() in {"O", "N", "F", "Cl", "Br", "I", "S", "P"}

def attached_h_count(atom) -> int:
    # with explicit Hs, count H neighbors
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "H")

def estimate_shift_for_protons(atom, mol) -> str:
    """
    Very rough rule-based 1H chemical shift estimate (ppm).
    Returns a range string.
    """
    if atom.GetSymbol() != "C":
        return "—"

    # Formyl H: carbonyl carbon carrying H
    if is_carbonyl_carbon(atom) and attached_h_count(atom) > 0:
        return "9.0–10.0 (aldehyde)"

    # Aromatic
    if atom.GetIsAromatic():
        return "6.0–8.5 (aromatic)"

    # Vinylic (sp2 carbon in C=C)
    if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
        # exclude carbonyl carbon already handled
        return "4.5–6.5 (vinylic)"

    # Check neighbors
    nbrs = list(atom.GetNeighbors())
    # directly attached to O/N/halogen
    if any(n.GetSymbol() in {"O"} for n in nbrs):
        return "3.2–4.2 (C–O)"
    if any(n.GetSymbol() in {"N"} for n in nbrs):
        return "2.5–4.0 (C–N)"
    if any(n.GetSymbol() in {"F", "Cl", "Br", "I"} for n in nbrs):
        return "3.0–4.5 (C–X)"

    # alpha to carbonyl
    if any(is_carbonyl_carbon(n) for n in nbrs):
        return "2.0–2.7 (α to C=O)"

    # benzylic (next to aromatic ring)
    if any(n.GetIsAromatic() for n in nbrs):
        return "2.2–3.0 (benzylic)"

    # simple alkyl
    h = attached_h_count(atom)
    # crude methyl vs methylene vs methine heuristic
    if h == 3:
        return "0.7–1.2 (alkyl CH₃)"
    if h == 2:
        return "1.0–1.6 (alkyl CH₂)"
    if h == 1:
        return "1.2–2.0 (alkyl CH)"
    return "0.8–2.0 (alkyl)"

def splitting_n_plus_1(atom, mol) -> str:
    """
    Approximate splitting via n+1 rule:
    n = total H on directly bonded carbon neighbors (ignoring hetero neighbors).
    """
    if atom.GetSymbol() != "C":
        return "—"

    n = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            continue
        if nbr.GetSymbol() == "C":
            n += attached_h_count(nbr)
        else:
            # hetero neighbors ignored (OH/NH exchange etc.)
            pass
    return multiplicity_name(n)

def group_proton_sets(mol):
    """
    Group proton-bearing heavy atoms by symmetry rank (approx).
    Returns list of dict rows.
    """
    # ranks for heavy atoms only (include Hs too, but we mainly want carbons with H)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rows = []

    # consider only carbon atoms that have >=1 H attached
    proton_carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == "C" and attached_h_count(a) > 0]

    # group by rank
    groups = {}
    for a in proton_carbons:
        r = ranks[a.GetIdx()]
        groups.setdefault(r, []).append(a)

    # build table rows
    for i, (rank, atoms) in enumerate(sorted(groups.items(), key=lambda x: x[0]), start=1):
        total_h = sum(attached_h_count(a) for a in atoms)
        # take representative atom for environment label
        rep = atoms[0]
        shift = estimate_shift_for_protons(rep, mol)
        split = splitting_n_plus_1(rep, mol)

        # quick environment note
        env = []
        if rep.GetIsAromatic():
            env.append("aromatic")
        if any(is_carbonyl_carbon(n) for n in rep.GetNeighbors()):
            env.append("α-C=O")
        if any(n.GetSymbol() == "O" for n in rep.GetNeighbors()):
            env.append("C–O")
        if any(n.GetSymbol() == "N" for n in rep.GetNeighbors()):
            env.append("C–N")
        if not env:
            env.append("alkyl/other")

        rows.append({
            "Group": f"H-group {i}",
            "Atoms in group": len(atoms),
            "H count (Integral)": int(total_h),
            "Shift δ (ppm)": shift,
            "Splitting (n+1)": split,
            "Intensity": int(total_h),
            "Notes": ", ".join(env),
        })

    return rows

# -----------------------
# 3) Compute + display
# -----------------------
mol = mol_from_smiles(smiles)
if mol is None:
    st.error("Diese SMILES konnte ich nicht parsen. Bitte nochmal zeichnen oder SMILES prüfen.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Molekül (Piktogramm)")
    img = Draw.MolToImage(Chem.RemoveHs(mol), size=(450, 450))
    st.image(img, caption=f"SMILES: {smiles}")

with col2:
    st.subheader("¹H-NMR Übungstabelle (regelbasiert)")
    rows = group_proton_sets(mol)
    if not rows:
        st.warning("Ich habe keine C–H-Protonen gefunden (oder das Molekül hat keine H an C).")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

st.caption(
    "Hinweis: Shift & Splitting sind hier **Heuristiken für Training**. "
    "In echten Spektren beeinflussen z.B. nicht-äquivalente Nachbarn, J-Werte, Konformation, Lösungsmittel usw. das Muster."
)
