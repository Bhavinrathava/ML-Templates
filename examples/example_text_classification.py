"""
Text Classification Example
============================

This script demonstrates how to use the TextClassifier for end-to-end
text classification: Text -> Embeddings -> Predictions.

Now supports TEXT LABELS that are automatically encoded/decoded!

Includes examples for:
1. Sentiment classification with text labels
2. Multi-class topic classification with text labels
3. Hierarchical text classification with DataFrame labels
4. Save/load model with label encoders

Requirements:
    pip install nlp-templates[torch,embeddings]

Usage:
    python examples/example_text_classification.py
"""

import numpy as np
import pandas as pd
from nlp_templates import TextClassifier, HierarchicalTextClassifier


def example_sentiment_classification_text_labels():
    """Example: Sentiment classification with TEXT LABELS."""
    print("=" * 70)
    print("Example 1: Sentiment Classification with TEXT LABELS")
    print("=" * 70)

    # Sample training data with TEXT LABELS (not numeric!)
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

    # TEXT LABELS - classifier will automatically encode these
    labels = [
        "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative",
    ]

    # Initialize text classifier
    print("\n[1] Initializing TextClassifier...")
    clf = TextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        classifier_type="simple",
        name="sentiment_classifier",
        test_size=0.25,
    )

    # Train - labels are automatically encoded
    print("\n[2] Training classifier with text labels...")
    clf.fit(
        texts=texts,
        labels=labels,  # TEXT LABELS!
        model_config={
            "type": "neural_network",
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 30,
                "batch_size": 4,
            },
        },
    )

    # Show detected classes
    print(f"\n    Detected classes: {clf.get_classes(level=0)}")

    # Evaluate
    print("\n[3] Evaluating...")
    results = clf.evaluate()
    print(f"    Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"    Test F1: {results['test_metrics']['f1']:.4f}")

    # Predict on new texts - returns ORIGINAL TEXT LABELS
    print("\n[4] Predicting on new texts (returns text labels)...")
    new_texts = [
        "This is the best thing I've bought this year!",
        "Complete garbage, threw it away immediately.",
        "It's okay, nothing special but works fine.",
    ]

    predictions = clf.predict(new_texts)  # Returns text labels!
    probabilities = clf.predict_proba(new_texts)

    for text, pred, prob in zip(new_texts, predictions, probabilities):
        print(f"\n    Text: '{text[:50]}...'")
        print(f"    Prediction: {pred}")  # Already a text label!
        print(f"    Confidence: {prob.max():.2%}")

    print("\n" + "=" * 70)
    return clf


def example_topic_classification_text_labels():
    """Example: Multi-class topic classification with TEXT LABELS."""
    print("\n" + "=" * 70)
    print("Example 2: Topic Classification with TEXT LABELS")
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

    # TEXT LABELS instead of numeric
    labels = [
        "Technology", "Technology", "Technology", "Technology",
        "Sports", "Sports", "Sports", "Sports",
        "Business", "Business", "Business", "Business",
        "Science", "Science", "Science", "Science",
    ]

    # Initialize and train
    print("\n[1] Initializing and training TextClassifier with text labels...")
    clf = TextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        name="topic_classifier",
        test_size=0.25,
    )

    clf.fit(
        texts=texts,
        labels=labels,  # TEXT LABELS!
        model_config={
            "type": "neural_network",
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 50,
                "batch_size": 4,
            },
        },
    )

    # Show detected classes
    print(f"\n    Detected classes: {clf.get_classes(level=0)}")

    # Evaluate
    print("\n[2] Evaluating...")
    results = clf.evaluate()
    print(f"    Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

    # Predict on new texts - returns TEXT LABELS
    print("\n[3] Predicting on new texts...")
    new_texts = [
        "Apple releases new MacBook with M3 chip",
        "Lakers defeat Celtics in overtime thriller",
        "Federal Reserve announces interest rate decision",
        "Scientists find water on distant exoplanet",
    ]

    predictions = clf.predict(new_texts)  # Returns text labels directly!

    for text, pred in zip(new_texts, predictions):
        print(f"    '{text[:45]}...' -> {pred}")

    print("\n" + "=" * 70)
    return clf


def example_hierarchical_with_dataframe():
    """Example: Hierarchical text classification with DataFrame labels."""
    print("\n" + "=" * 70)
    print("Example 3: Hierarchical Classification with DataFrame TEXT LABELS")
    print("=" * 70)

    # Sample product descriptions
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

    # Hierarchical TEXT LABELS as a DataFrame!
    labels = pd.DataFrame({
        "category": [
            "Electronics", "Electronics", "Electronics",
            "Electronics", "Electronics", "Electronics",
            "Electronics", "Electronics", "Electronics",
            "Clothing", "Clothing", "Clothing",
            "Clothing", "Clothing", "Clothing",
        ],
        "subcategory": [
            "Phones", "Phones", "Phones",
            "Laptops", "Laptops", "Laptops",
            "Accessories", "Accessories", "Accessories",
            "Shirts", "Shirts", "Shirts",
            "Pants", "Pants", "Pants",
        ],
    })

    print("\n    Label DataFrame:")
    print(labels.head(6).to_string(index=False))

    # Initialize hierarchical classifier
    print("\n[1] Initializing HierarchicalTextClassifier...")
    clf = HierarchicalTextClassifier(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        name="product_classifier",
        test_size=0.2,
    )

    # Train with DataFrame labels
    print("\n[2] Training hierarchical classifier with DataFrame labels...")
    clf.fit(
        texts=texts,
        labels=labels,  # DataFrame with text labels!
        model_config={
            "params": {
                "hidden_dims": [128, 64],
                "epochs": 50,
                "batch_size": 4,
            }
        },
    )

    # Show detected classes for each level
    print(f"\n    Category classes: {clf.get_classes(level=0)}")
    print(f"    Subcategory classes: {clf.get_classes(level=1)}")

    # Evaluate
    print("\n[3] Evaluating...")
    results = clf.evaluate()
    print("\n    Results by hierarchy level:")
    for level in range(2):
        level_key = f"level_{level}"
        if level_key in results["test_metrics"]:
            acc = results["test_metrics"][level_key]["accuracy"]
            print(f"    Level {level}: Accuracy = {acc:.4f}")

    # Predict on new texts - returns TEXT LABELS!
    print("\n[4] Predicting on new product descriptions...")
    new_texts = [
        "Smart watch with fitness tracking",
        "Wool sweater for cold weather",
        "Tablet with stylus pen support",
    ]

    # Returns text labels directly!
    predictions = clf.predict(new_texts)

    for text, pred in zip(new_texts, predictions):
        print(f"\n    Text: '{text}'")
        print(f"    Prediction: {pred}")  # Already text labels!

    # Use predict_as_dataframe for nice output
    print("\n[5] Predictions as DataFrame...")
    pred_df = clf.predict_as_dataframe(new_texts)
    print(pred_df.to_string(index=False))

    print("\n" + "=" * 70)
    return clf


def example_save_load_model():
    """Example: Save and load model with label encoders."""
    print("\n" + "=" * 70)
    print("Example 4: Save and Load Model")
    print("=" * 70)

    # Train a simple classifier
    texts = ["great!", "awesome!", "terrible!", "awful!"]
    labels = ["positive", "positive", "negative", "negative"]

    clf = TextClassifier(test_size=0.5)
    clf.fit(texts, labels)

    # Save the model
    import tempfile
    save_path = tempfile.mkdtemp()
    print(f"\n[1] Saving model to: {save_path}")
    clf.save(save_path)

    # Load the model
    print("\n[2] Loading model...")
    loaded_clf = TextClassifier.load(save_path)

    # Predict with loaded model
    print("\n[3] Predicting with loaded model...")
    new_predictions = loaded_clf.predict(["fantastic!", "horrible!"])
    print(f"    Predictions: {new_predictions}")

    print("\n" + "=" * 70)
    return loaded_clf


def example_embeddings_only():
    """Example: Get embeddings without classification."""
    print("\n" + "=" * 70)
    print("Example 5: Generate Embeddings Only")
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
    print("#" + " " * 12 + "TEXT CLASSIFICATION WITH TEXT LABELS" + " " * 13 + "#")
    print("#" * 70)

    # Run examples
    example_sentiment_classification_text_labels()
    example_topic_classification_text_labels()
    example_hierarchical_with_dataframe()
    example_save_load_model()
    example_embeddings_only()

    print("\n" + "#" * 70)
    print("#" + " " * 22 + "ALL EXAMPLES COMPLETE" + " " * 25 + "#")
    print("#" * 70)


if __name__ == "__main__":
    main()
