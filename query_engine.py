import re
from llama_index.core.query_engine import CustomQueryEngine
from config.settings import llm


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: any

    def custom_query(self, query_str: str) -> str:
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]
        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        prompt = (
            f"Given the following community summary:\n{community_summary}\n\n"
            f"Answer this query as best as possible: {query}"
        )
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        prompt = (
            "Combine the following partial answers into one concise, coherent response:\n\n"
            f"{community_answers}"
        )
        response = llm.generate_content(prompt)
        cleaned_final_response = response.text.strip()
        return cleaned_final_response
