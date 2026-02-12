"""
Home.py - Main entry point for the Spectroscopy Web Application.

This multi-page Streamlit application provides tools for molecular analysis
and NMR spectroscopy training, developed as part of a Bachelor's thesis at ZHAW.

Author: Leon Akryeziu
"""

import os
import sys

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


def _bootstrap_streamlit_when_run_as_python_script() -> None:
    """
    Re-launch via `streamlit run` when started with `python app.py`.
    This prevents the common "nothing opens" behavior in IDE run buttons.
    """
    if __name__ != "__main__":
        return
    if get_script_run_ctx() is not None:
        return

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    raise SystemExit(stcli.main())


_bootstrap_streamlit_when_run_as_python_script()

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Spectroscopy Tools",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E3A5F;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔬 Spectroscopy Analysis Tools</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">A scientific web application for molecular structure analysis and NMR spectroscopy</p>',
    unsafe_allow_html=True
)

st.divider()

# Introduction
st.markdown("""
## Welcome

This application provides interactive tools for molecular structure analysis and 
¹H-NMR spectroscopy training. It is designed for educational and research purposes
in the field of organic chemistry and spectroscopy.
""")

# Feature Overview
st.markdown("## Available Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🧬 SMILES Analyzer
    
    Analyze molecular structures from SMILES notation:
    
    - **SMILES Parsing**: Validate and canonicalize SMILES strings
    - **2D Visualization**: Generate publication-ready molecular structures
    - **Property Calculation**: Compute molecular formula, weight, and more
    - **Descriptor Analysis**: LogP, H-bond donors/acceptors, TPSA
    
    *Navigate to this tool using the sidebar →*
    """)

with col2:
    st.markdown("""
    ### 🧪 ¹H-NMR Trainer
    
    Learn and practice ¹H-NMR spectroscopy:
    
    - **Interactive Editor**: Draw molecules using the Ketcher editor
    - **NMR Prediction**: Rule-based chemical shift estimation
    - **Splitting Patterns**: Visualize n+1 coupling patterns
    - **Spectrum Simulation**: Generate simulated NMR spectra
    
    *Navigate to this tool using the sidebar →*
    """)

st.divider()

# Quick Start Guide
st.markdown("""
## Quick Start

1. **Select a tool** from the sidebar on the left
2. **Enter a SMILES string** or **draw a molecule** depending on the tool
3. **Analyze the results** displayed on the page

### Example SMILES Strings

| Molecule | SMILES |
|----------|--------|
| Ethanol | `CCO` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Caffeine | `Cn1cnc2c1c(=O)n(c(=O)n2C)C` |
| Ibuprofen | `CC(C)Cc1ccc(cc1)C(C)C(=O)O` |
""")

st.divider()

# Technical Information
with st.expander("ℹ️ Technical Information"):
    st.markdown("""
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **Cheminformatics**: RDKit
    - **Visualization**: Altair, RDKit Drawing
    - **Molecular Editor**: Ketcher (streamlit-ketcher)
    
    ### About
    
    This application was developed as part of a Bachelor's thesis at the 
    Zurich University of Applied Sciences (ZHAW).
    
    ### References
    
    - RDKit: Open-Source Cheminformatics Software
    - Streamlit: The fastest way to build data apps
    """)

# Footer
st.markdown("---")
st.caption("© 2026 ZHAW Bachelor Thesis - Spectroscopy Analysis Tools")
