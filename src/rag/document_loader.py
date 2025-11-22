from typing import List
import os

class DocumentLoader:
    def load_documents(self, paths: List[str]) -> List[str]:
        """
        Load documents from paths.
        """
        documents = []
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        return documents
