import streamlit as st
from modules.loader import load_document
from modules.splitter import get_nodes
from modules.extractor import create_extractor
from modules.graph_store import GraphRAGStore
from modules.query_engine import GraphRAGQueryEngine
from config.settings import llm
from llama_index.core import PropertyGraphIndex

st.set_page_config(page_title="GraphRAG Document Explorer", layout="wide")

st.title("🕸️ GraphRAG Document Explorer")

uploaded_file = st.file_uploader("Upload your document (CSV, TXT, or PDF)", type=["csv", "txt", "pdf"])

if uploaded_file:
    st.info("Processing document...")
    documents = load_document(uploaded_file)
    nodes = get_nodes(documents)

    # ✅ pass llm to the extractor
    extractor = create_extractor(llm)
    graph_store = GraphRAGStore()

    with st.spinner("Building Knowledge Graph..."):
        index = PropertyGraphIndex(
            nodes=nodes,
            property_graph_store=graph_store,
            kg_extractors=[extractor],  # uses async acall() under the hood
            show_progress=True,
        )
        index.property_graph_store.build_communities()

    st.success("✅ Knowledge Graph built successfully!")

    query = st.text_input("Ask something about your document:")
    if st.button("Query"):
        if query.strip():
            query_engine = GraphRAGQueryEngine(graph_store=index.property_graph_store, llm=llm)
            response = query_engine.query(query)
            st.markdown(f"### 🧠 Response\n{response.response}")
        else:
            st.warning("Please enter a query.")
