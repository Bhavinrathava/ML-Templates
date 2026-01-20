"""
Text Classification Example
============================

This script demonstrates how to use the TextClassifier for end-to-end
text classification: Text -> Embeddings -> Predictions.

Includes examples for:
1. Simple sentiment classification
2. Multi-class topic classification
3. Hierarchical text classification

Requirements:
    pip install nlp-templates[torch]
    pip install sentence-transformers

Usage:
    python examples/example_text_classification.py
"""

import numpy as np
from nlp_templates import TextClassifier, HierarchicalTextClassifier


def example_sentiment_classification():
    """Example: Binary sentiment classification."""
    print("=" * 70)
    print("Example 1: Sentiment Classification (Binary)")
    print("=" * 70)

    # Sample training data
    texts = [
        # Positive reviews
        "This product is amazing! Best purchase I've ever made.",
        "Excellent quality and fast shipping. Highly recommended!",
        "Love it! Works perfectly and looks great.",
        "Outstanding service and beautiful product.",
        "Fantastic experience, will buy again!",
        "Super happy with this purchase, exceeded expectations.",
        "Great value for money, couldn't be happier.",
        "Perfect! Exactly what I was looking for.",
        # Negative reviews
        "Terrible quality. Broke after one day.",
        "Waste of money. Don't buy this.",
        "Very disappointed. Nothing like the description.",
        "Poor customer service and defective product.",
        "Awful experience. Requested a refund.",
        "Cheaply made and doesn't work properly.",
        "Not worth it. Save your money.",
        "Horrible. One star is too generous.",
    ]

    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

    # Initialize text classifier
    print("\n[1] Initializing TextClassifier...")
    clf = TextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        classifier_type="simple",
        name="sentiment_classifier",
        test_size=0.25,
    )

    # Train
    print("\n[2] Training classifier...")
    clf.fit(
        texts=texts,
        labels=labels,
        model_config={
            "type": "neural_network",
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 30,
                "batch_size": 4,
            },
        },
    )

    # Evaluate
    print("\n[3] Evaluating...")
    results = clf.evaluate()
    print(f"    Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"    Test F1: {results['test_metrics']['f1']:.4f}")

    # Predict on new texts
    print("\n[4] Predicting on new texts...")
    new_texts = [
        "This is the best thing I've bought this year!",
        "Complete garbage, threw it away immediately.",
        "It's okay, nothing special but works fine.",
    ]

    predictions = clf.predict(new_texts)
    probabilities = clf.predict_proba(new_texts)

    sentiment_map = {0: "Negative", 1: "Positive"}
    for text, pred, prob in zip(new_texts, predictions, probabilities):
        print(f"\n    Text: '{text[:50]}...'")
        print(f"    Prediction: {sentiment_map[pred]}")
        print(f"    Confidence: {prob.max():.2%}")

    print("\n" + "=" * 70)
    return clf


def example_topic_classification():
    """Example: Multi-class topic classification."""
    print("\n" + "=" * 70)
    print("Example 2: Topic Classification (Multi-class)")
    print("=" * 70)

    # Sample data for 4 topics
    texts = [
        # Technology
        "New smartphone features advanced AI capabilities",
        "Software update improves system performance",
        "Cloud computing transforms enterprise infrastructure",
        "Cybersecurity threats increase in digital age",
        # Sports
        "Team wins championship after incredible season",
        "Star player signs record-breaking contract",
        "Olympic athletes prepare for upcoming games",
        "Soccer match ends in thrilling penalty shootout",
        # Business
        "Stock market reaches all-time high today",
        "Startup raises millions in funding round",
        "CEO announces company restructuring plans",
        "Economic forecast predicts steady growth",
        # Science
        "Researchers discover new species in rainforest",
        "Space telescope captures distant galaxy images",
        "Climate study reveals alarming temperature trends",
        "Medical breakthrough offers hope for patients",
    ]

    # Labels: 0=Technology, 1=Sports, 2=Business, 3=Science
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    # Initialize and train
    print("\n[1] Initializing and training TextClassifier...")
    clf = TextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        name="topic_classifier",
        test_size=0.25,
    )

    clf.fit(
        texts=texts,
        labels=labels,
        model_config={
            "type": "neural_network",
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 50,
                "batch_size": 4,
            },
        },
    )

    # Evaluate
    print("\n[2] Evaluating...")
    results = clf.evaluate()
    print(f"    Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

    # Predict on new texts
    print("\n[3] Predicting on new texts...")
    new_texts = [
        "Apple releases new MacBook with M3 chip",
        "Lakers defeat Celtics in overtime thriller",
        "Federal Reserve announces interest rate decision",
        "Scientists find water on distant exoplanet",
    ]

    predictions = clf.predict(new_texts)
    topic_map = {0: "Technology", 1: "Sports", 2: "Business", 3: "Science"}

    for text, pred in zip(new_texts, predictions):
        print(f"    '{text[:45]}...' -> {topic_map[pred]}")

    print("\n" + "=" * 70)
    return clf


def example_hierarchical_classification():
    """Example: Hierarchical text classification."""
    print("\n" + "=" * 70)
    print("Example 3: Hierarchical Text Classification")
    print("=" * 70)

    # Sample data with 2-level hierarchy
    # Level 0: Category (Electronics=0, Clothing=1)
    # Level 1: Subcategory
    #   - Electronics: Phones=0, Laptops=1, Accessories=2
    #   - Clothing: Shirts=0, Pants=1

    texts = [
        # Electronics - Phones
        "Latest iPhone with amazing camera features",
        "Android smartphone with 5G connectivity",
        "Budget phone with long battery life",
        # Electronics - Laptops
        "Gaming laptop with RTX graphics card",
        "Ultrabook perfect for business travel",
        "MacBook Pro for professional video editing",
        # Electronics - Accessories
        "Wireless earbuds with noise cancellation",
        "USB-C hub with multiple ports",
        "Portable charger for all devices",
        # Clothing - Shirts
        "Cotton t-shirt comfortable for summer",
        "Formal dress shirt for office wear",
        "Casual polo shirt in various colors",
        # Clothing - Pants
        "Denim jeans classic fit style",
        "Comfortable sweatpants for relaxing",
        "Formal trousers for business meetings",
    ]

    # Hierarchical labels: [Category, Subcategory]
    labels = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],  # Electronics - Phones
            [0, 1],
            [0, 1],
            [0, 1],  # Electronics - Laptops
            [0, 2],
            [0, 2],
            [0, 2],  # Electronics - Accessories
            [1, 0],
            [1, 0],
            [1, 0],  # Clothing - Shirts
            [1, 1],
            [1, 1],
            [1, 1],  # Clothing - Pants
        ]
    )

    # Initialize hierarchical classifier
    print("\n[1] Initializing HierarchicalTextClassifier...")
    clf = HierarchicalTextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        name="product_classifier",
        test_size=0.2,
    )

    # Train
    print("\n[2] Training hierarchical classifier...")
    clf.fit(
        texts=texts,
        labels=labels,
        model_config={
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 50,
                "batch_size": 4,
            }
        },
    )

    # Evaluate
    print("\n[3] Evaluating...")
    results = clf.evaluate()
    print("\n    Results by hierarchy level:")
    for level in range(2):
        level_key = f"level_{level}"
        if level_key in results["test_metrics"]:
            acc = results["test_metrics"][level_key]["accuracy"]
            print(f"    Level {level}: Accuracy = {acc:.4f}")

    # Predict on new texts
    print("\n[4] Predicting on new product descriptions...")
    new_texts = [
        "Smart watch with fitness tracking",
        "Wool sweater for cold weather",
        "Tablet with stylus pen support",
    ]

    predictions = clf.predict(new_texts)

    category_map = {0: "Electronics", 1: "Clothing"}
    subcat_map = {
        (0, 0): "Phones",
        (0, 1): "Laptops",
        (0, 2): "Accessories",
        (1, 0): "Shirts",
        (1, 1): "Pants",
    }

    for text, pred in zip(new_texts, predictions):
        cat = category_map.get(pred[0], f"Unknown({pred[0]})")
        subcat = subcat_map.get(tuple(pred), f"Unknown({pred[1]})")
        print(f"\n    Text: '{text}'")
        print(f"    Category: {cat} -> Subcategory: {subcat}")

    print("\n" + "=" * 70)
    return clf


def example_embeddings_only():
    """Example: Get embeddings without classification."""
    print("\n" + "=" * 70)
    print("Example 4: Generate Embeddings Only")
    print("=" * 70)

    # Initialize classifier (embedder is lazily loaded)
    clf = TextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    texts = [
        "Machine learning is fascinating",
        "Deep learning revolutionizes AI",
        "I love pizza and pasta",
    ]

    print("\n[1] Generating embeddings for texts...")
    embeddings = clf.get_embeddings(texts, show_progress=False)

    print(f"    Number of texts: {len(texts)}")
    print(f"    Embedding shape: {embeddings.shape}")
    print(f"    Embedding dimension: {embeddings.shape[1]}")

    # Show similarity between texts
    print("\n[2] Computing cosine similarity between texts...")
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix = cosine_similarity(embeddings)

    print("\n    Similarity matrix:")
    for i, t1 in enumerate(texts):
        for j, t2 in enumerate(texts):
            if i < j:
                print(f"    '{t1[:30]}...' <-> '{t2[:30]}...': {sim_matrix[i, j]:.4f}")

    print("\n" + "=" * 70)


def main():
    """Run all examples."""
    print("\n" + "#" * 70)
    print("#" + " " * 18 + "TEXT CLASSIFICATION EXAMPLES" + " " * 20 + "#")
    print("#" * 70)

    # Run examples
    example_sentiment_classification()
    example_topic_classification()
    example_hierarchical_classification()
    example_embeddings_only()

    print("\n" + "#" * 70)
    print("#" + " " * 22 + "ALL EXAMPLES COMPLETE" + " " * 25 + "#")
    print("#" * 70)


if __name__ == "__main__":
    main()
