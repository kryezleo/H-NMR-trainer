"""
H-NMR Trainer - Interactive ¹H-NMR spectroscopy training tool.

This module provides functionality to:
- Draw molecules using the Ketcher editor
- Predict chemical shifts based on structural rules
- Visualize splitting patterns and simulated spectra

Part of the Spectroscopy Analysis Tools for Bachelor's Thesis at ZHAW.
"""

import re
from math import comb

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
try:
    from streamlit_ketcher import st_ketcher
except Exception:
    st_ketcher = None

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

# Page configuration
st.set_page_config(
    page_title="H-NMR Trainer",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def mol_from_smiles(smiles_str: str) -> Chem.Mol:
    """
    Parse SMILES and return molecule with explicit hydrogens.
    
    Args:
        smiles_str: Input SMILES string
        
    Returns:
        RDKit Mol object with explicit Hs, or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    return mol


def multiplicity_name(n: int) -> str:
    """
    Convert number of coupling neighbors to multiplicity name.
    
    Args:
        n: Number of equivalent neighboring protons
        
    Returns:
        Multiplicity name string
    """
    return {
        0: "singlet",
        1: "doublet",
        2: "triplet",
        3: "quartet",
        4: "quintet",
        5: "sextet",
        6: "septet",
    }.get(n, f"{n+1}-plet")


def is_carbonyl_carbon(atom) -> bool:
    """Check if atom is a carbonyl carbon (C=O)."""
    if atom.GetSymbol() != "C":
        return False
    for b in atom.GetBonds():
        if b.GetBondType() == BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetSymbol() == "O":
                return True
    return False


def is_hetero(atom) -> bool:
    """Check if atom is a heteroatom."""
    return atom.GetSymbol() in {"O", "N", "F", "Cl", "Br", "I", "S", "P"}


def attached_h_count(atom) -> int:
    """Count explicitly attached hydrogen atoms."""
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "H")


def estimate_shift_for_protons(atom, mol) -> str:
    """
    Estimate ¹H chemical shift based on structural environment.
    
    Uses rule-based heuristics for educational purposes.
    Returns a range string with environment description.
    
    Args:
        atom: Carbon atom bearing the protons
        mol: Parent molecule
        
    Returns:
        Shift range string (e.g., "0.7–1.2 (alkyl CH₃)")
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
        return "4.5–6.5 (vinylic)"

    # Check neighbors
    nbrs = list(atom.GetNeighbors())
    
    # Directly attached to O/N/halogen
    if any(n.GetSymbol() in {"O"} for n in nbrs):
        return "3.2–4.2 (C–O)"
    if any(n.GetSymbol() in {"N"} for n in nbrs):
        return "2.5–4.0 (C–N)"
    if any(n.GetSymbol() in {"F", "Cl", "Br", "I"} for n in nbrs):
        return "3.0–4.5 (C–X)"

    # Alpha to carbonyl
    if any(is_carbonyl_carbon(n) for n in nbrs):
        return "2.0–2.7 (α to C=O)"

    # Benzylic (next to aromatic ring)
    if any(n.GetIsAromatic() for n in nbrs):
        return "2.2–3.0 (benzylic)"

    # Simple alkyl - differentiate by H count
    h = attached_h_count(atom)
    if h == 3:
        return "0.7–1.2 (alkyl CH₃)"
    if h == 2:
        return "1.0–1.6 (alkyl CH₂)"
    if h == 1:
        return "1.2–2.0 (alkyl CH)"
    return "0.8–2.0 (alkyl)"


def splitting_n_plus_1(atom, mol) -> str:
    """
    Calculate splitting pattern using the n+1 rule.
    
    Args:
        atom: Carbon atom bearing the protons
        mol: Parent molecule
        
    Returns:
        Multiplicity name string
    """
    if atom.GetSymbol() != "C":
        return "—"

    n = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            continue
        if nbr.GetSymbol() == "C":
            n += attached_h_count(nbr)
    return multiplicity_name(n)


def group_proton_sets(mol) -> list:
    """
    Group proton-bearing carbons by symmetry and calculate NMR parameters.
    
    Args:
        mol: RDKit Mol object with explicit Hs
        
    Returns:
        List of dictionaries containing NMR data for each group
    """
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rows = []

    # Consider only carbon atoms that have >=1 H attached
    proton_carbons = [
        a for a in mol.GetAtoms() 
        if a.GetSymbol() == "C" and attached_h_count(a) > 0
    ]

    # Group by symmetry rank
    groups = {}
    for a in proton_carbons:
        r = ranks[a.GetIdx()]
        groups.setdefault(r, []).append(a)

    # Build table rows
    for i, (rank, atoms) in enumerate(sorted(groups.items(), key=lambda x: x[0]), start=1):
        total_h = sum(attached_h_count(a) for a in atoms)
        rep = atoms[0]  # Representative atom
        shift = estimate_shift_for_protons(rep, mol)
        split = splitting_n_plus_1(rep, mol)

        # Environment notes
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


def parse_shift_range(shift_str: str) -> float:
    """Extract middle value from shift range string."""
    match = re.search(r'([\d.]+)[–-]([\d.]+)', shift_str)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return (low + high) / 2
    return 1.0


def generate_nmr_spectrum(rows, x_min=0, x_max=12, num_points=2000):
    """
    Generate a simulated ¹H-NMR spectrum with Lorentzian peaks.
    
    Args:
        rows: List of proton group data
        x_min: Minimum ppm value
        x_max: Maximum ppm value
        num_points: Number of data points
        
    Returns:
        Tuple of (x, y) arrays for the spectrum
    """
    x = np.linspace(x_min, x_max, num_points)
    y = np.zeros_like(x)
    
    width = 0.04  # Peak width for visualization
    
    total_h = sum(row["H count (Integral)"] for row in rows)
    if total_h == 0:
        return x, y
    
    for row in rows:
        shift_str = row["Shift δ (ppm)"]
        h_count = row["H count (Integral)"]
        center = parse_shift_range(shift_str)
        
        # Splitting pattern
        split = row["Splitting (n+1)"]
        n_peaks = 1
        if "doublet" in split:
            n_peaks = 2
        elif "triplet" in split:
            n_peaks = 3
        elif "quartet" in split:
            n_peaks = 4
        elif "quintet" in split:
            n_peaks = 5
        elif "sextet" in split:
            n_peaks = 6
        elif "septet" in split:
            n_peaks = 7
        
        J = 0.025  # J coupling constant (ppm spacing)
        
        # Pascal's triangle coefficients for multiplet intensities
        pascal_coeffs = [comb(n_peaks - 1, i) for i in range(n_peaks)]
        pascal_sum = sum(pascal_coeffs)
        
        # Generate multiplet pattern
        for i in range(n_peaks):
            peak_fraction = pascal_coeffs[i] / pascal_sum
            peak_intensity = h_count * peak_fraction
            peak_pos = center + (i - (n_peaks - 1) / 2) * J
            # Lorentzian peak shape
            y += peak_intensity * (width ** 2) / ((x - peak_pos) ** 2 + width ** 2)
    
    # Scale for proper display
    max_h = max(row["H count (Integral)"] for row in rows)
    if y.max() > 0:
        y = y * (max_h / y.max())
    
    return x, y


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

st.title("🧪 ¹H-NMR Trainer")
st.markdown("""
Draw a molecule in the editor below, click **Apply**, and get:
- A **molecular structure image**
- A **table** with: **H-groups**, **shift (ppm)**, **splitting**, **integral**
- A **simulated NMR spectrum**
""")

st.divider()

# Ketcher Editor
st.subheader("Molecular Editor")
smiles = ""

if st_ketcher is not None:
    st.caption("⬇️ Draw your molecule in the editor below. Click **Apply** (top right) when done.")
    smiles = st_ketcher(height=500)
else:
    st.warning("Ketcher editor unavailable in this session. Using SMILES input fallback.")
    smiles = st.text_input(
        "Enter SMILES:",
        value="CCO",
        placeholder="e.g. CCO"
    )

# Display current SMILES
if smiles:
    st.success(f"**SMILES:** `{smiles}`")

if not smiles or not isinstance(smiles, str) or len(smiles.strip()) == 0:
    st.info("Draw a molecule and click **Apply** (top right in the editor).")
    st.stop()

# Parse molecule
mol = mol_from_smiles(smiles)
if mol is None:
    st.error("Could not parse this SMILES. Please redraw or check the structure.")
    st.stop()

st.divider()

# Results display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Molecular Structure")
    img = Draw.MolToImage(Chem.RemoveHs(mol), size=(450, 450))
    st.image(img, caption=f"SMILES: {smiles}")

with col2:
    st.subheader("¹H-NMR Data Table (Rule-Based)")
    rows = group_proton_sets(mol)
    if not rows:
        st.warning("No C–H protons found in this molecule.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

# NMR Spectrum
st.divider()
st.subheader("Simulated ¹H-NMR Spectrum")

if rows:
    x, y = generate_nmr_spectrum(rows)
    max_h = max(row["H count (Integral)"] for row in rows)
    
    spectrum_df = pd.DataFrame({
        'ppm': x,
        'Intensity': y
    })
    
    chart = alt.Chart(spectrum_df).mark_line(
        color='#1f77b4', 
        strokeWidth=1.5
    ).encode(
        x=alt.X('ppm:Q', 
                scale=alt.Scale(domain=[12, 0]),  # Inverted x-axis (NMR convention)
                title='Chemical Shift δ (ppm)'),
        y=alt.Y('Intensity:Q', 
                title='Relative Intensity (H count)',
                scale=alt.Scale(domain=[0, max_h * 1.15]))
    ).properties(
        height=350,
        title='Simulated ¹H-NMR Spectrum'
    ).configure_axis(
        grid=True,
        gridColor='#e0e0e0'
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Integration info
    st.caption("**Integrals (area ∝ H count):**")
    peak_info = []
    for row in rows:
        shift = parse_shift_range(row["Shift δ (ppm)"])
        peak_info.append(
            f"δ {shift:.1f} ppm: **{row['H count (Integral)']}H** ({row['Splitting (n+1)']})"
        )
    st.write(" | ".join(peak_info))
else:
    st.info("No protons found for spectrum generation.")

st.caption(
    "**Note:** Shifts and splitting patterns are **rule-based heuristics for training purposes**. "
    "Real spectra are influenced by conformations, solvent effects, coupling constants, and other factors."
)

# Footer
st.markdown("---")
st.caption("¹H-NMR Trainer - Part of Spectroscopy Analysis Tools")
