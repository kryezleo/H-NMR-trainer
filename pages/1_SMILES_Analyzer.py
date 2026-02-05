"""
SMILES Analyzer - Molecular structure analysis from SMILES notation.

This module provides functionality to:
- Parse and canonicalize SMILES strings using RDKit
- Display 2D molecular structures
- Compute and display molecular properties

Part of the Spectroscopy Analysis Tools for Bachelor's Thesis at ZHAW.
"""

import streamlit as st
from io import BytesIO
from typing import Optional, Dict, Any
import requests

from rdkit import Chem
from rdkit.Chem import (
    Draw,
    Descriptors,
    rdMolDescriptors,
    AllChem,
)
from rdkit.Chem.Draw import rdMolDraw2D

# Page configuration
st.set_page_config(
    page_title="SMILES Analyzer",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES string and return an RDKit Mol object.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        RDKit Mol object if valid, None otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return None
    
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to its canonical form.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES string if valid, None otherwise
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    
    return Chem.MolToSmiles(mol, canonical=True)


def calculate_molecular_formula(mol: Chem.Mol) -> str:
    """
    Calculate the molecular formula from an RDKit Mol object.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Molecular formula as string
    """
    return rdMolDescriptors.CalcMolFormula(mol)


def calculate_molecular_properties(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Calculate various molecular properties.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary containing molecular properties
    """
    properties = {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 4),
        "Exact Mass (g/mol)": round(Descriptors.ExactMolWt(mol), 6),
        "Heavy Atom Count": Descriptors.HeavyAtomCount(mol),
        "Number of Atoms": mol.GetNumAtoms(),
        "Number of Bonds": mol.GetNumBonds(),
        "Number of Rings": rdMolDescriptors.CalcNumRings(mol),
        "Number of Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Number of Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "LogP (Wildman-Crippen)": round(Descriptors.MolLogP(mol), 4),
        "TPSA (Å²)": round(Descriptors.TPSA(mol), 2),
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Fraction Csp³": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
    }
    return properties


def generate_2d_image(mol: Chem.Mol, size: tuple = (500, 500)) -> bytes:
    """
    Generate a 2D molecular structure image.
    
    Args:
        mol: RDKit Mol object
        size: Image dimensions (width, height)
        
    Returns:
        PNG image as bytes
    """
    from io import BytesIO
    
    # Compute 2D coordinates if not present
    AllChem.Compute2DCoords(mol)
    
    # Use Draw.MolToImage which is more reliable across environments
    img = Draw.MolToImage(mol, size=size)
    
    # Convert PIL Image to bytes
    img_buffer = BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return img_buffer.getvalue()


def generate_svg_image(mol: Chem.Mol, size: tuple = (500, 500)) -> str:
    """
    Generate a 2D molecular structure as SVG.
    
    Args:
        mol: RDKit Mol object
        size: Image dimensions (width, height)
        
    Returns:
        SVG as string
    """
    # Compute 2D coordinates if not present
    AllChem.Compute2DCoords(mol)
    
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def check_lipinski_rules(properties: Dict[str, Any]) -> Dict[str, bool]:
    """
    Check Lipinski's Rule of Five for drug-likeness.
    
    Args:
        properties: Dictionary of molecular properties
        
    Returns:
        Dictionary with rule compliance status
    """
    rules = {
        "MW ≤ 500": properties["Molecular Weight (g/mol)"] <= 500,
        "LogP ≤ 5": properties["LogP (Wildman-Crippen)"] <= 5,
        "HBD ≤ 5": properties["H-Bond Donors"] <= 5,
        "HBA ≤ 10": properties["H-Bond Acceptors"] <= 10,
    }
    return rules


@st.cache_data(ttl=3600)
def get_molecule_name_from_pubchem(smiles: str) -> Optional[str]:
    """
    Retrieve the IUPAC name or common name of a molecule from PubChem.
    
    Args:
        smiles: Canonical SMILES string
        
    Returns:
        Molecule name if found, None otherwise
    """
    try:
        # First, get the CID from SMILES
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/cids/JSON"
        response = requests.get(url, timeout=3)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        cid = data.get("IdentifierList", {}).get("CID", [None])[0]
        
        if not cid:
            return None
        
        # Get the compound properties including name
        props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,Title/JSON"
        props_response = requests.get(props_url, timeout=3)
        
        if props_response.status_code != 200:
            return None
        
        props_data = props_response.json()
        properties = props_data.get("PropertyTable", {}).get("Properties", [{}])[0]
        
        # Prefer Title (common name) over IUPAC name
        name = properties.get("Title") or properties.get("IUPACName")
        return name
    
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_pubchem_synonyms(smiles: str, max_synonyms: int = 5) -> list:
    """
    Retrieve synonyms/alternative names from PubChem.
    
    Args:
        smiles: Canonical SMILES string
        max_synonyms: Maximum number of synonyms to return
        
    Returns:
        List of synonym strings
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/synonyms/JSON"
        response = requests.get(url, timeout=3)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
        
        return synonyms[:max_synonyms]
    
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

st.title("🧬 SMILES Analyzer")
st.markdown("""
Analyze molecular structures from SMILES (Simplified Molecular-Input Line-Entry System) notation.
Enter a SMILES string to parse, visualize, and calculate molecular properties.
""")

st.divider()

# Input Section
st.subheader("Input")

# Example SMILES for convenience
example_smiles = {
    "Select an example...": "",
    "Ethanol": "CCO",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "Benzene": "c1ccccc1",
    "Cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
}

col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g., CCO for ethanol",
        help="Enter a valid SMILES string to analyze the molecular structure"
    )

with col_input2:
    selected_example = st.selectbox(
        "Or choose an example:",
        options=list(example_smiles.keys())
    )

# Use example if selected, otherwise use manual input
smiles = example_smiles.get(selected_example, "") if selected_example != "Select an example..." else smiles_input
if not smiles:
    smiles = smiles_input

# Analysis
if smiles:
    mol = parse_smiles(smiles)
    
    if mol is None:
        st.error(f"❌ Invalid SMILES: `{smiles}`")
        st.info("Please check your SMILES string and try again.")
        st.stop()
    
    # Canonical SMILES
    canonical = canonicalize_smiles(smiles)
    
    st.success(f"✅ Valid SMILES parsed successfully")
    
    st.divider()
    
    # Results in columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("2D Molecular Structure")
        
        # Generate and display structure
        try:
            img_bytes = generate_2d_image(mol, size=(450, 450))
            st.image(img_bytes, caption=canonical)
            
            # Download button for image
            st.download_button(
                label="📥 Download Structure (PNG)",
                data=img_bytes,
                file_name="molecule_structure.png",
                mime="image/png"
            )
        except Exception as e:
            # Fallback to simpler rendering
            img = Draw.MolToImage(mol, size=(450, 450))
            st.image(img, caption=f"Structure of: {canonical}")
    
    with col2:
        st.subheader("Molecule Information")
        
        # Optional: Lookup molecule name from PubChem
        if st.button("🔍 Lookup Name from PubChem"):
            with st.spinner("Searching PubChem..."):
                molecule_name = get_molecule_name_from_pubchem(canonical)
                synonyms = get_pubchem_synonyms(canonical, max_synonyms=5)
                
                if molecule_name:
                    st.markdown(f"### {molecule_name}")
                    if synonyms:
                        st.markdown("**Alternative Names:**")
                        st.caption(", ".join(synonyms))
                else:
                    st.info("Molecule not found in PubChem database")
        
        st.markdown("**Input SMILES:**")
        st.code(smiles, language=None)
        
        st.markdown("**Canonical SMILES:**")
        st.code(canonical, language=None)
        
        # InChI and InChIKey
        try:
            inchi = Chem.MolToInchi(mol)
            inchi_key = Chem.MolToInchiKey(mol)
            
            st.markdown("**InChI:**")
            st.code(inchi, language=None)
            
            st.markdown("**InChIKey:**")
            st.code(inchi_key, language=None)
        except Exception:
            st.info("InChI generation not available")
    
    st.divider()
    
    # Molecular Properties
    st.subheader("Molecular Properties")
    
    properties = calculate_molecular_properties(mol)
    
    # Display in organized columns
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("**Basic Properties**")
        st.metric("Molecular Formula", properties["Molecular Formula"])
        st.metric("Molecular Weight", f"{properties['Molecular Weight (g/mol)']:.2f} g/mol")
        st.metric("Exact Mass", f"{properties['Exact Mass (g/mol)']:.4f} g/mol")
        st.metric("Heavy Atoms", properties["Heavy Atom Count"])
    
    with col_p2:
        st.markdown("**Structural Features**")
        st.metric("Total Atoms", properties["Number of Atoms"])
        st.metric("Total Bonds", properties["Number of Bonds"])
        st.metric("Rings", properties["Number of Rings"])
        st.metric("Aromatic Rings", properties["Number of Aromatic Rings"])
        st.metric("Rotatable Bonds", properties["Number of Rotatable Bonds"])
    
    with col_p3:
        st.markdown("**Physicochemical Properties**")
        st.metric("LogP", f"{properties['LogP (Wildman-Crippen)']:.2f}")
        st.metric("TPSA", f"{properties['TPSA (Å²)']:.1f} Ų")
        st.metric("H-Bond Donors", properties["H-Bond Donors"])
        st.metric("H-Bond Acceptors", properties["H-Bond Acceptors"])
        st.metric("Fraction Csp³", f"{properties['Fraction Csp³']:.2f}")
    
    st.divider()
    
    # Lipinski's Rule of Five
    st.subheader("Drug-Likeness (Lipinski's Rule of Five)")
    
    lipinski = check_lipinski_rules(properties)
    violations = sum(1 for v in lipinski.values() if not v)
    
    if violations == 0:
        st.success("✅ Molecule follows all Lipinski rules (drug-like)")
    elif violations <= 1:
        st.warning(f"⚠️ {violations} Lipinski rule violation(s)")
    else:
        st.error(f"❌ {violations} Lipinski rule violations")
    
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    
    for col, (rule, passes) in zip([col_l1, col_l2, col_l3, col_l4], lipinski.items()):
        with col:
            status = "✅" if passes else "❌"
            st.markdown(f"{status} **{rule}**")
    
    st.divider()
    
    # Full properties table
    with st.expander("📊 View All Properties as Table"):
        import pandas as pd
        df = pd.DataFrame(list(properties.items()), columns=["Property", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Enter a SMILES string above or select an example to begin analysis.")

# Footer
st.markdown("---")
st.caption("SMILES Analyzer - Part of Spectroscopy Analysis Tools")
