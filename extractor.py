import requests
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import numpy as np
from typing import Dict, List, Tuple
import re
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dataclasses import dataclass
import json


@dataclass
class ProductAttribute:
    text: str
    confidence: float
    source_element: str
    context: str = ""

class GenericProductExtractor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enhanced semantic concepts for e-commercew
        self.semantic_concepts = {
            'title': [
                "This is the main product name",
                "This is what the product is called",
                "This is the product title",
                "This is the item name for sale",
                "Ceci est le nom principal du produit",
                "Ceci est le nom sous lequel le produit est vendu",
                "Ceci est le titre du produit",
                "Ceci est le nom de l'article à vendre"
            ],
            'description': [
                "This describes the product features and details",
                "This explains what the product is about",
                "This gives product specifications",
                "This is the product description",
                "These are the product details",
                "Ceci décrit les caractéristiques et les détails du produit",
                "Ceci explique ce qu'est le produit",
                "Cela donne les spécifications du produit",
                "Ceci est la description du produit",
                "Ce sont les détails du produit"
            ],
            'price': [
                "This is how much the product costs",
                "This is the product price",
                "This shows the amount to pay",
                "This is the current price",
                "This is the sale price",
                "C'est le coût du produit",
                "Ceci est le prix du produit",
                "Ceci montre le montant à payer",
                "Ceci est le prix actuel",
                "Ceci est le prix en promotion"
            ],
            'image': [
                "This shows what the product looks like",
                "This is a product photo",
                "This is the main product image",
                "This is the product gallery",
                "Ceci montre à quoi ressemble le produit",
                "Ceci est une photo du produit",
                "Ceci est l'image principale du produit",
                "Ceci est la galerie de produits"
            ],
            'size': [
                "These are the available sizes",
                "This shows what sizes you can choose",
                "These are the size options",
                "This is the size selection",
                "Voici les tailles disponibles",
                "Ceci montre les tailles que vous pouvez choisir",
                "Ce sont les options de taille",
                "Ceci est la sélection de taille"
            ],
            'color': [
                "These are the available colors",
                "This shows what colors you can choose",
                "These are the color options",
                "This is the color selection",
                "Voici les couleurs disponibles",
                "Ceci montre les couleurs que vous pouvez choisir",
                "Ce sont les options de couleur",
                "Ceci est la sélection de couleur"
            ],
            'sku': [
                "This is the product SKU",
                "This is the product reference number",
                "This is the product code",
                "This is the item number",
                "Ceci est le SKU du produit",
                "Ceci est le numéro de référence du produit",
                "Ceci est le code du produit",
                "Ceci est le numéro de l'article"
            ]
        }

        
        # Pre-compute embeddings for semantic concepts
        self.concept_embeddings = {
            category: self.model.encode(concepts)
            for category, concepts in self.semantic_concepts.items()
        }

        # Common color patterns
        self.color_patterns = [
            r'\b(?:black|white|red|blue|green|yellow|purple|brown|gray|grey|pink|orange|navy|beige|cream|gold|silver)\b',
            r'\b(?:dark|light|pale|bright|deep)\s+(?:black|white|red|blue|green|yellow|purple|brown|gray|grey|pink|orange|navy|beige|cream|gold|silver)\b'
        ]

        # Common size patterns
        self.size_patterns = [
            r'\b(?:XXS|XS|S|M|L|XL|XXL|XXXL)\b',
            r'\b(?:small|medium|large|extra\s+small|extra\s+large)\b',
            r'\b\d+(?:\.\d+)?\s*(?:cm|mm|in|")\b',
            r'\b(?:one size|universal|free size)\b',
            r'\b(?:\d{2}/\d{2}|\d{2}-\d{2})\b'  # European sizes like 36/38
        ]

    def _extract_structured_elements(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all potentially relevant elements with their context."""
        elements = []
        
        # Define priority elements and their weights
        priority_tags = {
            'h1': 0.9, 'h2': 0.8, 'h3': 0.7, 'title': 0.9,
            'p': 0.6, 'span': 0.5, 'div': 0.5, 'li': 0.4,
            'meta': 0.7, 'section': 0.5  # Add more tags
        }


        def get_element_context(element):
            """Extract contextual information from element and its parents"""
            contexts = []
            for parent in element.parents:
                class_name = parent.get('class', [])
                id_name = parent.get('id', '')
                if class_name:
                    contexts.extend(class_name)
                if id_name:
                    contexts.append(id_name)
            return ' '.join(contexts)

        # Extract text-based elements
        for tag, base_weight in priority_tags.items():
            for element in soup.find_all(tag):
                text = element.get_text(strip=True)
                if text and len(text) > 1:  # Filter out single characters
                    context = get_element_context(element)
                    elements.append({
                        'text': text,
                        'tag': tag,
                        'weight': base_weight,
                        'context': context,
                        'type': 'text',
                        'attributes': dict(element.attrs)
                    })

        # Extract images
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                context = get_element_context(img)
                elements.append({
                    'text': alt,
                    'tag': 'img',
                    'weight': 0.8,
                    'context': context,
                    'type': 'image',
                    'src': src,
                    'attributes': dict(img.attrs)
                })

        # Extract structured data if available
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    elements.append({
                        'text': json.dumps(data),
                        'tag': 'structured',
                        'weight': 1.0,
                        'context': 'json-ld',
                        'type': 'structured',
                        'data': data
                    })
            except:
                continue

        return elements


    def _extract_json_ld(self, soup: BeautifulSoup) -> Dict:
        """Extract JSON-LD product data."""
        json_ld_data = {}
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    data = data[0]  # Take first item if it's a list
                if isinstance(data, dict):
                    if '@type' in data and data['@type'] in ['Product', 'IndividualProduct']:
                        json_ld_data = data
                        break
            except:
                continue
        return json_ld_data

    def _extract_meta_data(self, soup: BeautifulSoup) -> Dict:
        """Extract product data from meta tags."""
        meta_data = {}
        meta_mappings = {
            'og:title': 'title',
            'og:description': 'description',
            'og:image': 'image',
            'product:price:amount': 'price',
            'product:brand': 'brand',
            'product:color': 'color',
            'product:size': 'size'
        }
        
        for meta in soup.find_all('meta'):
            property_name = meta.get('property', '')
            content = meta.get('content', '')
            if property_name in meta_mappings and content:
                meta_data[meta_mappings[property_name]] = content
        
        return meta_data

    def _extract_colors(self, text: str) -> List[str]:
        """Extract color information from text."""
        colors = set()
        for pattern in self.color_patterns:
            matches = re.finditer(pattern, text.lower())
            colors.update(match.group(0) for match in matches)
        return list(colors)

    def _extract_sizes(self, text: str) -> List[str]:
        """Extract size information from text."""
        sizes = set()
        for pattern in self.size_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            sizes.update(match.group(0) for match in matches)
        return list(sizes)
    

    def _semantic_similarity(self, text_embedding: np.ndarray, concept_embeddings: np.ndarray) -> float:
        """Calculate semantic similarity between text and concept embeddings."""
        similarities = [np.dot(text_embedding, concept_emb) / 
                       (np.linalg.norm(text_embedding) * np.linalg.norm(concept_emb))
                       for concept_emb in concept_embeddings]
        return max(similarities)

    def _classify_element(self, element: Dict, text_embedding: np.ndarray) -> Tuple[str, float]:
        """Classify an element's content type based on semantic similarity."""
        best_category = None
        best_score = -1

        for category, concept_embs in self.concept_embeddings.items():
            score = self._semantic_similarity(text_embedding, concept_embs)
            score *= element['weight']  # Apply HTML context weight
            
            if score > best_score:
                best_score = score
                best_category = category

        return best_category, best_score
    def extract_info(self, html: str) -> Dict[str, ProductAttribute]:

        soup = BeautifulSoup(html, 'html.parser')
        
        # Initialize results
        results = {}
        
        # Extract structured data first
        json_ld_data = self._extract_json_ld(soup)
        print(json_ld_data)
        meta_data = self._extract_meta_data(soup)
        print(meta_data)
        
        # Combine structured data
        if json_ld_data:
            if 'name' in json_ld_data:
                results['title'] = ProductAttribute(
                    json_ld_data['name'], 1.0, 'structured-data', 'json-ld')
            if 'description' in json_ld_data:
                results['description'] = ProductAttribute(
                    json_ld_data['description'], 1.0, 'structured-data', 'json-ld')
            if 'offers' in json_ld_data:
                offers = json_ld_data['offers']
                if isinstance(offers, dict) and 'price' in offers:
                    results['price'] = ProductAttribute(
                        str(offers['price']), 1.0, 'structured-data', 'json-ld')
        
        # Add meta data
        for key, value in meta_data.items():
            if key not in results:
                results[key] = ProductAttribute(value, 0.9, 'meta-tags', 'meta')
        
        # Extract regular elements
        elements = self._extract_structured_elements(soup)
        print(elements)
        # Process text elements
        text_elements = [e for e in elements if e['type'] == 'text']
        if text_elements:
            text_embeddings = self.model.encode([e['text'] for e in text_elements])
            
            # Combine all text for color and size extraction
            all_text = ' '.join(e['text'] for e in text_elements)
            colors = self._extract_colors(all_text)
            sizes = self._extract_sizes(all_text)
            
            if colors:
                results['colors'] = ProductAttribute(
                    ', '.join(colors), 0.8, 'text-analysis', 'color-extraction')
            
            if sizes:
                results['sizes'] = ProductAttribute(
                    ', '.join(sizes), 0.8, 'text-analysis', 'size-extraction')

            for element, embedding in zip(text_elements, text_embeddings):
                category, score = self._classify_element(element, embedding)
                
                if score > 0.5:
                    if category == 'price':
                        prices = self._extract_price_candidates(element['text'])
                        if prices:
                            element['text'] = prices[0]
                    
                    if category not in results or score > results[category].confidence:
                        results[category] = ProductAttribute(
                            element['text'],
                            score,
                            element['tag'],
                            element['context']
                        )


            # Process images
            images = [e for e in elements if e['type'] == 'image']
            if images:
                # Sort images by their context and attributes
                def image_score(img):
                    score = img['weight']
                    if any(kw in img['text'].lower() for kw in ['product', 'main', 'primary']):
                        score += 0.3
                    if any(kw in img['context'].lower() for kw in ['product', 'main', 'primary']):
                        score += 0.2
                    return score

                best_image = max(images, key=image_score)
                results['image'] = ProductAttribute(
                    best_image['src'],
                    image_score(best_image),
                    'img',
                    best_image['text']  # Use alt text as context
                )


        
        return results

def main():
    extractor = GenericProductExtractor()
    
    url = "https://www.kiabi.be/fr/jean-slim-a-5-poches-l34-bleu_P830455C852780"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    response = requests.get(url, headers=headers)
    if(response.status_code != 200):
        raise  Exception("Failed to retrieve page")

    html = response.text
    results = extractor.extract_info(html)
    
    print("\nExtracted Information:")
    for category, attr in results.items():
        print(f"\n{category.upper()}:")
        print(f"Text: {attr.text}")
        print(f"Confidence: {attr.confidence:.2f}")
        print(f"Source: {attr.source_element}")
        print(f"Context: {attr.context}")

if __name__ == "__main__":
    main()