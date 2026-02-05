"""
H-NMR.py - Legacy entry point, redirects to new multi-page structure.
Please use: streamlit run Home.py
"""

import streamlit as st

st.set_page_config(page_title="Redirect", layout="wide")

st.markdown("## ⚠️ Diese Datei wurde verschoben")
st.info("""
Die App wurde zu einer Multi-Page-Struktur umgebaut.

**Lokal starten mit:**
```bash
streamlit run Home.py
```
""")

st.markdown("### Verfügbare Seiten:")
st.page_link("Home.py", label="🔬 Home", icon="🏠")
st.page_link("pages/1_SMILES_Analyzer.py", label="🧬 SMILES Analyzer")
st.page_link("pages/2_H-NMR_Trainer.py", label="🧪 H-NMR Trainer")
