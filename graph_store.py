import re
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.graph_stores import SimplePropertyGraphStore
from config.settings import llm


class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using Gemini."""
        prompt = (
            "You are provided with a set of relationships from a knowledge graph, each represented as "
            "entity1->entity2->relation->relationship_description. Create a concise summary capturing key entities and "
            "their relationships, focusing on relevance and coherence.\n\n"
            f"Relationships:\n{text}"
        )
        response = llm.generate_content(prompt)
        clean_response = response.text.strip()
        return clean_response

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info)

    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties.get("relationship_description", ""),
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        community_mapping = {item.node: item.cluster for item in clusters}
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster
            node = item.node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id:
                    edge_data = nx_graph.get_edge_data(node, neighbor)
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail)
        return community_info

    def _summarize_communities(self, community_info):
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
