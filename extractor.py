import asyncio
import re
import nest_asyncio
from typing import Any, Callable, List
from llama_index.core.schema import TransformComponent, BaseNode
from pydantic import Field

nest_asyncio.apply()

entity_pattern = r'entity_name:\s*(.+?)\s*entity_type:\s*(.+?)\s*entity_description:\s*(.+?)\s*'
relationship_pattern = r'source_entity:\s*(.+?)\s*target_entity:\s*(.+?)\s*relation:\s*(.+?)\s*relationship_description:\s*(.+?)\s*'

def parse_fn(response_str: str):
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, extract entities and relationships in the following format:
entity_name: ...
entity_type: ...
entity_description: ...
source_entity: ...
target_entity: ...
relation: ...
relationship_description: ...
"""

class GraphRAGExtractor(TransformComponent):
    llm: Any = Field(...)
    extract_prompt: str = Field(...)
    max_paths_per_chunk: int = Field(default=2)
    parse_fn: Callable = Field(default=parse_fn)

    async def acall(self, nodes: List[BaseNode], **kwargs):
        results = []
        for node in nodes:
            text = node.get_content()
            prompt = f"{self.extract_prompt}\n\n{text}"

            # ✅ Use LlamaIndex-compatible LLM call
            response = await asyncio.to_thread(self.llm.complete, prompt)

            # Ollama in LlamaIndex returns an object with `.text`
            output_text = getattr(response, "text", str(response))

            entities, relationships = self.parse_fn(output_text)
            node.metadata["entities"] = entities
            node.metadata["relationships"] = relationships
            results.append(node)
        return results

    def __call__(self, nodes: List[BaseNode], **kwargs):
        return asyncio.run(self.acall(nodes, **kwargs))


def create_extractor(llm):
    return GraphRAGExtractor(
        llm=llm,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=2,
        parse_fn=parse_fn,
    )
