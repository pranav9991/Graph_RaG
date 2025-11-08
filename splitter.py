from llama_index.core.node_parser import SentenceSplitter

def get_nodes(documents, chunk_size=1024, chunk_overlap=20):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes
