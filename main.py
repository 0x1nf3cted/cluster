
# import asyncio
# import json
# from transformers import pipeline
# from typing import List, Dict, Any
# import numpy as np
# import torch
# from datasets import Dataset  # Import the Dataset class from the datasets library
# from urllib.parse import urlparse

# class ShoppingLinkClassifier:
#     def __init__(self):
#         # Initialize the classifier upon instantiation
#         asyncio.run(self.initialize_classifier())

#     async def initialize_classifier(self):
#         try:
#             # Check if CUDA is available and select the device accordingly
#             device = 0 if torch.cuda.is_available() else -1
#             # Initialize the pipeline with GPU support if available
#             self.classifier = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=device)
#             print(f"Classifier initialized successfully on {'GPU' if device == 0 else 'CPU'}.")
#         except Exception as error:
#             print("Error initializing classifier:", error)
#             raise error

#     async def get_embedding(self, texts: List[str]) -> List[List[float]]:
#         try:
#             # Get embeddings for the provided texts in batch
#             outputs = await asyncio.to_thread(self.classifier, texts)
#             embeddings = [np.mean(output[0], axis=0) for output in outputs]  # Mean pooling over the token embeddings
#             # Normalize embeddings
#             norm_embeddings = [embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) != 0 else embedding
#                                for embedding in embeddings]
#             return norm_embeddings
#         except Exception as error:
#             print("Error getting embeddings:", error)
#             raise error

#     @staticmethod
#     def cosine_similarity(a: List[float], b: List[float]) -> float:
#         dot_product = sum(x * y for x, y in zip(a, b))
#         magnitude_a = sum(x ** 2 for x in a) ** 0.5
#         magnitude_b = sum(y ** 2 for y in b) ** 0.5
#         return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0

#     @staticmethod
#     def extract_relevant_text(url: str) -> str:
#         parsed_url = urlparse(url)
#         relevant_parts = [
#             part.replace('-', ' ').replace('_', ' ') for part in parsed_url.path.split('/')
#         ] + list(parsed_url.query.split('&'))
#         return ' '.join(part for part in relevant_parts if part).lower()

#     async def classify_link(self, url: str, url_embedding: List[float]) -> Dict[str, Any]:
#         categories = ["women", "men", "kids", "accessories"]
#         category_descriptions = {
#             "women": [
#                 "women's clothing", "women's fashion", "ladies' clothes", "feminine clothes",
#                 "women's collection", "women's style", "clothes for women", "vêtements pour femmes", "femmes"
#             ],
#             "men": [
#                 "men's clothing", "men's fashion", "men's clothes", "masculine clothes",
#                 "men's collection", "men's style", "clothes for men", "vêtements pour hommes", "hommes",
#             ],
#             "kids": [
#                 "children's clothing", "kids' fashion", "baby clothes", "youth clothes",
#                 "kids' collection", "teen clothes", "clothes for kids", "vêtements pour enfants", "enfant", "bébé",
#             ],
#             "accessories": [
#                 "fashion accessories", "jewelry", "bags", "shoes", "belts", "scarves", "hats",
#                 "fashion items", "articles de mode",
#             ],
#         }

#         highest_similarity = -1
#         best_category = "unknown"

#         for category in categories:
#             category_embeddings = await asyncio.gather(
#                 *[self.get_embedding([desc]) for desc in category_descriptions[category]]
#             )

#             # Calculate average similarity across all descriptions
#             avg_similarity = sum(
#                 self.cosine_similarity(url_embedding, embedding[0]) for embedding in category_embeddings
#             ) / len(category_embeddings)

#             if avg_similarity > highest_similarity:
#                 highest_similarity = avg_similarity
#                 best_category = category

#         confidence_threshold = 0.4
#         final_category = best_category if highest_similarity >= confidence_threshold else "unknown"

#         return {
#             "url": url,
#             "category": final_category,
#             "confidence": highest_similarity,
#         }

#     async def batch_classify_links(self, urls: List[str]) -> List[Dict[str, Any]]:
#         # Prepare data for batch processing with datasets
#         texts = [self.extract_relevant_text(url) for url in urls]
#         dataset = Dataset.from_dict({"url": urls, "text": texts})
        
#         # Get embeddings for all URLs in a single batch
#         embeddings = await self.get_embedding(dataset["text"])
        
#         results = []
#         for i, url in enumerate(urls):
#             url_embedding = embeddings[i]
#             result = await self.classify_link(url, url_embedding)  # Use precomputed embeddings
#             results.append(result)
        
#         return results

# def retrieve_links(json_string):
#     # Parse the JSON data
#     data = json.loads(json_string)
    
#     # Initialize a list to store links
#     links = []
    
#     # Loop through the categories in the JSON data
#     for category, items in data.items():
#         for item in items:
#             # Extract the link and append it to the list
#             link = item.get("link")
#             if link:
#                 links.append(link)
    
#     return links


# # Example usage
# if __name__ == "__main__":
#     classifier = ShoppingLinkClassifier()

#     with open('links.json', 'r') as file:
#         json_data = file.read()

#     links = retrieve_links(json_data)

#     results = asyncio.run(classifier.batch_classify_links(links))
#     for result in results:
#         print(f"Result: {result}")

import asyncio
import json
from transformers import pipeline
from typing import List, Dict, Any
import numpy as np
import torch
from datasets import Dataset
from urllib.parse import urlparse

class ShoppingLinkClassifier:
    def __init__(self):
        asyncio.run(self.initialize_classifier())

    async def initialize_classifier(self):
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=device)
            print(f"Classifier initialized on {'GPU' if device == 0 else 'CPU'}.")
        except Exception as error:
            print("Error initializing classifier:", error)
            raise error

    async def get_embedding(self, texts: List[str]) -> List[List[float]]:
        try:
            outputs = await asyncio.to_thread(self.classifier, texts)
            embeddings = [np.mean(output[0], axis=0) for output in outputs]
            norm_embeddings = [embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) != 0 else embedding
                               for embedding in embeddings]
            return norm_embeddings
        except Exception as error:
            print("Error getting embeddings:", error)
            raise error

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x ** 2 for x in a) ** 0.5
        magnitude_b = sum(y ** 2 for y in b) ** 0.5
        return dot_product / (magnitude_a * magnitude_b) if magnitude_a and magnitude_b else 0

    @staticmethod
    def extract_relevant_text(url: str) -> str:
        parsed_url = urlparse(url)
        relevant_parts = [
            part.replace('-', ' ').replace('_', ' ') for part in parsed_url.path.split('/')
        ] + list(parsed_url.query.split('&'))
        return ' '.join(part for part in relevant_parts if part).lower()

    async def classify_link(self, url: str, url_embedding: List[float]) -> Dict[str, Any]:
        categories = ["women", "men", "kids", "accessories"]
        category_descriptions = {
            "women": [
                "women's clothing", "women's fashion", "ladies' clothes", "feminine clothes",
                "women's collection", "women's style", "clothes for women", "vêtements pour femmes", "femmes"
            ],
            "men": [
                "men's clothing", "men's fashion", "men's clothes", "masculine clothes",
                "men's collection", "men's style", "clothes for men", "vêtements pour hommes", "hommes",
            ],
            "kids": [
                "children's clothing", "kids' fashion", "baby clothes", "youth clothes",
                "kids' collection", "teen clothes", "clothes for kids", "vêtements pour enfants", "enfant", "bébé",
            ],
            "accessories": [
                "fashion accessories", "jewelry", "bags", "shoes", "belts", "scarves", "hats",
                "fashion items", "articles de mode",
            ],
        }

        highest_similarity = -1
        best_category = "unknown"

        for category in categories:
            category_embeddings = await asyncio.gather(
                *[self.get_embedding([desc]) for desc in category_descriptions[category]]
            )
            avg_embedding = np.mean([embedding[0] for embedding in category_embeddings], axis=0)
            avg_similarity = self.cosine_similarity(url_embedding, avg_embedding)

            if avg_similarity > highest_similarity:
                highest_similarity = avg_similarity
                best_category = category

        confidence_threshold = 0.3
        final_category = best_category if highest_similarity >= confidence_threshold else "unknown"

        return {
            "url": url,
            "category": final_category,
            "confidence": highest_similarity,
        }

    async def batch_classify_links(self, urls: List[str]) -> List[Dict[str, Any]]:
        texts = [self.extract_relevant_text(url) for url in urls]
        dataset = Dataset.from_dict({"url": urls, "text": texts})
        
        # Obtain and normalize embeddings in one go
        embeddings = await self.get_embedding(dataset["text"])
        
        results = []
        for i, url in enumerate(urls):
            url_embedding = embeddings[i]
            result = await self.classify_link(url, url_embedding)
            results.append(result)
        
        return results

def retrieve_links(json_string):
    data = json.loads(json_string)
    links = []
    for category, items in data.items():
        for item in items:
            link = item.get("link")
            if link:
                links.append(link)
    return links

if __name__ == "__main__":
    classifier = ShoppingLinkClassifier()

    with open('output.json', 'r') as file:
        json_data = file.read()

    links = retrieve_links(json_data)

    results = asyncio.run(classifier.batch_classify_links(links))
    for result in results:
        print(f"Result: {result}")
