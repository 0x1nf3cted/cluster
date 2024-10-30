#
# AUteur: 0x1nf3cted
# Date: October 30, 2024
# Description:  ça fait du clustering (homme, femme, autre, ect...)




import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import unquote
import re

def load_data(file_path):
    """Load URLs data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_url(url):
    """Extract meaningful text from URL."""
    url = unquote(url)
    url = re.sub(r'https?://|www\.|\.com|\.fr|[0-9]|[^\w\s]', ' ', url)
    return ' '.join(url.lower().split())

def preprocess_title(title):
    """Clean and preprocess title text."""
    if not title:
        return ""
    title = re.sub(r'[^\w\s]', ' ', title.lower())
    return ' '.join(title.split())

class URLClassifier:
    def __init__(self, categories, category_descriptions):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.categories = categories
        self.category_descriptions = category_descriptions
        
        self.category_embeddings = {}
        for category, descriptions in self.category_descriptions.items():
            embeddings = self.model.encode(descriptions)
            self.category_embeddings[category] = np.mean(embeddings, axis=0)

    def classify_url(self, title, link):
        """Classify a single URL based on its title and link."""
        processed_title = preprocess_title(title)
        processed_url = preprocess_url(link)
        
        text_to_encode = f"{processed_title} {processed_url}".strip()
        if not text_to_encode:
            return "unknown"
        
        url_embedding = self.model.encode([text_to_encode])[0]
        
        similarities = {}
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                url_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            similarities[category] = similarity
        
        best_category = max(similarities.items(), key=lambda x: x[1])
        
        if best_category[1] > 0.15:   
            return best_category[0]
        return "unknown"

    def classify_urls(self, urls_data):
        """Classify multiple URLs and return results."""
        results = {}
        for website, urls in urls_data.items():
            website_results = []
            for url_info in urls:
                classification = self.classify_url(
                    url_info.get('title', ''),
                    url_info.get('link', '')
                )
                website_results.append({
                    'title': url_info.get('title', ''),
                    'link': url_info.get('link', ''),
                    'category': classification
                })
            results[website] = website_results
        return results



def main():
    # Initialize categories and descriptions
    categories = ["women", "men", "kids", "accessories"]
    category_descriptions = {
        "women": [
            "women's clothing", "women's fashion", "ladies' clothes", "feminine clothes",
            "women's collection", "women's style", "clothes for women", "vêtements pour femmes", "femmes"
        ],
        "men": [
            "men's clothing", "men's fashion", "men's clothes", "masculine clothes",
            "men's collection", "men's style", "clothes for men", "vêtements pour hommes", "hommes"
        ],
        "kids": [
            "children's clothing", "kids' fashion", "baby clothes", "youth clothes",
            "kids' collection", "teen clothes", "clothes for kids", "vêtements pour enfants", "enfant", "bébé"
        ],
        "accessories": [
            "fashion accessories", "jewelry", "bags", "shoes", "belts", "scarves", "hats",
            "fashion items", "articles de mode"
        ]
    }

    urls_data = load_data('links.json')

    classifier = URLClassifier(categories, category_descriptions)
    results = classifier.classify_urls(urls_data)

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()