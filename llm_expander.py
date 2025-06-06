from abc import abstractmethod
import pyterrier as pt
import torch
import pandas as pd
import re
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LLMExpander(pt.transformer.Transformer):
    def __init__(self, model_name: str, device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.max_new_tokens = 128

    @abstractmethod
    def build_prompt(self, query: str) -> str:
        pass

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        expanded_records = []
        for _, row in topics.iterrows():
            qid = row["qid"]
            original_query = row["query"]
            prompt = self.build_prompt(original_query)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            llm_output = re.sub(rf"[{re.escape(string.punctuation)}]", " ", llm_output)
            expanded_query = " ".join([original_query] * 5) + " " + llm_output
            expanded_records.append({"qid": qid, "query": expanded_query})
        return pd.DataFrame(expanded_records)


class Q2DZSExpander(LLMExpander):
    def build_prompt(self, query: str) -> str:
        return f"Write a passage that answers the following query: {query}"


class Q2EZSExpander(LLMExpander):
    def build_prompt(self, query: str) -> str:
        return f"Write a list of keywords for the following query: {query}"


class CoTExpander(LLMExpander):
    def build_prompt(self, query: str) -> str:
        return f"""
Answer the following query: {query}
Give the rationale before answering
"""


class Q2EFSExpander(LLMExpander):
    def build_prompt(self, query: str) -> str:
        return f"""
Write a list of additional keywords for a search query
Write the list as one string with spaces between. Don't include duplicates of keywords. Don't include the query itself.

Here are some examples:
Query: where was jaws filmed amity
Keywords: martha's vineyard filming locations movie set shooting site jaws location beach massachusetts island movie scene harbor ocean town

Query: what is the mass of a beta
Keywords: beta particle electron mass subatomic particle physics neutron decay radiation energy charge

Query: what is beef burgundy
Keywords: beef bourguignon recipe wine stew french dish ingredients cooking red wine braised meat

Query: difference between affiliate and subsidiary
Keywords: business structure ownership control corporation company legal entity parent company partnership relationship

Query: does candida cause anxiety
Keywords: candida overgrowth gut brain axis mental health yeast infection microbiome symptoms mood depression

Now it is your turn:
Query: {query}
Keywords: 
"""
