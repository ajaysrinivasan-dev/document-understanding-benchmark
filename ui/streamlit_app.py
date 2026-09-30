from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Document Understanding Benchmark", layout="wide")
st.title("Document Understanding Benchmark")
uploaded = st.file_uploader("Document", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"])
pipeline = st.selectbox("Pipeline", ["ocr", "vlm", "both"])
if uploaded and st.button("Extract", type="primary"):
    endpoint = os.getenv("API_URL", "http://localhost:8000") + "/extract"
    response = requests.post(
        endpoint,
        params={"pipeline": pipeline},
        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
        timeout=180,
    )
    if response.ok:
        payload = response.json()
        st.metric("Processing time", f"{payload['processing_time_ms']:.1f} ms")
        for result in payload["results"]:
            st.subheader(result["pipeline"].upper())
            st.json(result["fields"])
            for table in result["tables"]:
                st.dataframe(table["rows"], use_container_width=True)
            if result["warnings"]:
                st.warning("; ".join(result["warnings"]))
            if result["errors"]:
                st.error("; ".join(result["errors"]))
    else:
        st.error(f"Request failed ({response.status_code})")
