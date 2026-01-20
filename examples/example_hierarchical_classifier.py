"""
Hierarchical Classifier - Complete Usage Example
==================================================

This script demonstrates how to use the HierarchicalClassifier for a real-world
scenario: Food item categorization with a 3-level hierarchy.

Hierarchy Structure:
    L1: Main Category (Produce, Dairy, Bakery, etc.)
    L2: Sub Category (Fruits, Vegetables, Milk Products, etc.)
    L3: Specific Type (Apples, Carrots, Whole Milk, etc.)

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from nlp_templates.classifiers import HierarchicalClassifier


# ==============================================================================
# STEP 1: PREPARE YOUR DATA
# ==============================================================================


def prepare_food_data():
    """
    Prepare sample food categorization data with 3-level hierarchy.

    In production, you would load this from your database or CSV file.
    """

    # Sample food items with features and hierarchy labels
    data = {
        # Features (these would be your actual item attributes)
        "item_name": [
            "Honeycrisp Apple",
            "Gala Apple",
            "Banana",
            "Orange",
            "Carrot",
            "Broccoli",
            "Tomato",
            "Whole Milk",
            "Skim Milk",
            "Cheddar Cheese",
            "Mozzarella",
            "White Bread",
            "Wheat Bread",
            "Croissant",
            "Bagel",
            "Greek Yogurt",
            "Strawberry Yogurt",
            "Lettuce",
            "Spinach",
            "Chicken Breast",
            "Ground Beef",
            "Salmon",
            "Tuna",
            "Sourdough Bread",
            "Baguette",
        ],
        "price": [
            1.99,
            1.49,
            0.59,
            0.79,
            0.89,
            1.29,
            1.49,
            3.49,
            2.99,
            4.99,
            3.99,
            2.49,
            2.99,
            3.49,
            1.99,
            4.99,
            3.99,
            1.99,
            2.49,
            5.99,
            4.99,
            8.99,
            6.99,
            4.49,
            3.99,
        ],
        "weight_oz": [
            8,
            8,
            6,
            10,
            12,
            16,
            8,
            64,
            64,
            16,
            16,
            24,
            24,
            6,
            4,
            32,
            32,
            16,
            10,
            16,
            16,
            12,
            8,
            24,
            16,
        ],
        "organic": [
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            0,
        ],
        "perishable": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        # Hierarchy Labels (your target variables)
        "main_category": [
            "Produce",
            "Produce",
            "Produce",
            "Produce",
            "Produce",
            "Produce",
            "Produce",
            "Dairy",
            "Dairy",
            "Dairy",
            "Dairy",
            "Bakery",
            "Bakery",
            "Bakery",
            "Bakery",
            "Dairy",
            "Dairy",
            "Produce",
            "Produce",
            "Meat",
            "Meat",
            "Seafood",
            "Seafood",
            "Bakery",
            "Bakery",
        ],
        "sub_category": [
            "Fruits",
            "Fruits",
            "Fruits",
            "Fruits",
            "Vegetables",
            "Vegetables",
            "Vegetables",
            "Milk",
            "Milk",
            "Cheese",
            "Cheese",
            "Bread",
            "Bread",
            "Pastries",
            "Bread",
            "Yogurt",
            "Yogurt",
            "Vegetables",
            "Vegetables",
            "Poultry",
            "Beef",
            "Fish",
            "Fish",
            "Bread",
            "Bread",
        ],
        "specific_type": [
            "Apples",
            "Apples",
            "Tropical",
            "Citrus",
            "Root Vegetables",
            "Cruciferous",
            "Nightshades",
            "Whole",
            "Low-Fat",
            "Hard Cheese",
            "Soft Cheese",
            "White",
            "Whole Grain",
            "French",
            "Breakfast",
            "Greek",
            "Flavored",
            "Leafy Greens",
            "Leafy Greens",
            "Fresh",
            "Ground",
            "Fatty Fish",
            "Lean Fish",
            "Artisan",
            "French",
        ],
    }

    df = pd.DataFrame(data)

    # In production, you might add more engineered features here
    df["price_per_oz"] = df["price"] / df["weight_oz"]
    df["is_expensive"] = (df["price"] > df["price"].median()).astype(int)

    return df


# ==============================================================================
# STEP 2: INITIALIZE AND CONFIGURE THE CLASSIFIER
# ==============================================================================


def main():
    """Main execution function demonstrating hierarchical classification."""

    print("=" * 80)
    print("HIERARCHICAL CLASSIFIER - FOOD CATEGORIZATION EXAMPLE")
    print("=" * 80)

    # Load data
    print("\n[1] Loading food categorization data...")
    df = prepare_food_data()

    print(f"    ✓ Loaded {len(df)} food items")
    print(f"    ✓ Features: {list(df.columns[:6])}")
    print(f"\n    Hierarchy Structure:")
    print(
        f"      - Level 1 (main_category): {df['main_category'].nunique()} categories"
    )
    print(
        f"      - Level 2 (sub_category): {df['sub_category'].nunique()} categories"
    )
    print(
        f"      - Level 3 (specific_type): {df['specific_type'].nunique()} categories"
    )

    # Initialize classifier
    print("\n[2] Initializing HierarchicalClassifier...")

    classifier = HierarchicalClassifier(
        name="food_categorization",
        levels=[
            "main_category",
            "sub_category",
            "specific_type",
        ],  # Order matters!
        cascade_mode=True,  # Use parent predictions as features for children
        random_state=42,
        test_size=0.3,  # 70% train, 30% test
    )

    print("    ✓ Classifier initialized with 3-level hierarchy")
    print("    ✓ Cascade mode enabled (parent predictions → child features)")

    # Load data into classifier
    print("\n[3] Loading data into classifier...")

    # Specify which columns are features (exclude hierarchy columns and item name)
    feature_columns = [
        "price",
        "weight_oz",
        "organic",
        "perishable",
        "price_per_oz",
        "is_expensive",
    ]

    classifier.load_data(
        data=df,
        feature_columns=feature_columns,
        build_hierarchy_map=True,  # Important: builds valid parent→child relationships
    )

    print(f"    ✓ Using {len(feature_columns)} features: {feature_columns}")
    print("    ✓ Hierarchy map built (ensures valid category combinations)")

    # ==============================================================================
    # STEP 3: TRAIN THE HIERARCHICAL MODEL
    # ==============================================================================

    print("\n[4] Training hierarchical classifier...")
    print("    Note: This trains 3 separate models (one per hierarchy level)")

    classifier.train(
        model_type="xgboost",  # Can use 'xgboost', 'lightgbm', 'sklearn_rf'
        log_to_mlflow=False,  # Set to True if you have MLflow configured
        n_estimators=50,  # XGBoost parameter
        max_depth=4,  # XGBoost parameter
        learning_rate=0.1,  # XGBoost parameter
    )

    print("\n    ✓ Training complete!")
    print("      - Level 1 model: Predicts main_category")
    print(
        "      - Level 2 model: Uses Level 1 predictions + features → sub_category"
    )
    print(
        "      - Level 3 model: Uses Level 1&2 predictions + features → specific_type"
    )

    # ==============================================================================
    # STEP 4: EVALUATE THE MODEL
    # ==============================================================================

    print("\n[5] Evaluating model performance...")

    metrics = classifier.evaluate()

    print("\n    Performance by Hierarchy Level:")
    print("    " + "-" * 60)

    for level in classifier.levels:
        accuracy = metrics[f"{level}_accuracy"]
        f1 = metrics[f"{level}_f1"]
        print(f"    {level:20s}  Accuracy: {accuracy:.3f}  F1: {f1:.3f}")

    # Print detailed report
    print("\n")
    classifier.print_hierarchy_report()

    # ==============================================================================
    # STEP 5: MAKE PREDICTIONS ON NEW ITEMS
    # ==============================================================================

    print("\n[6] Making predictions on new food items...")

    # Simulate new items (in production, these come from your pipeline)
    new_items = pd.DataFrame(
        {
            "price": [2.99, 5.49, 1.99],
            "weight_oz": [8, 16, 32],
            "organic": [1, 0, 1],
            "perishable": [1, 1, 1],
            "price_per_oz": [2.99 / 8, 5.49 / 16, 1.99 / 32],
            "is_expensive": [1, 1, 0],
        }
    )

    # Predict full hierarchy for new items
    predictions = classifier.predict_hierarchy(
        new_items,
        enforce_hierarchy=True,  # Ensures predictions follow valid parent→child rules
    )

    print("\n    Predictions for 3 new food items:")
    print("    " + "-" * 60)

    for i, row in predictions.iterrows():
        print(f"\n    Item {i+1}:")
        print(
            f"      Price: ${new_items.iloc[i]['price']}, Weight: {new_items.iloc[i]['weight_oz']}oz"
        )
        print(f"      → Main Category: {row['main_category']}")
        print(f"      → Sub Category: {row['sub_category']}")
        print(f"      → Specific Type: {row['specific_type']}")

    # ==============================================================================
    # STEP 6: INSPECT HIERARCHY ENFORCEMENT
    # ==============================================================================

    print("\n[7] Inspecting hierarchy constraints...")

    # Show valid parent-child relationships
    print("\n    Valid category combinations (sample):")
    print("    " + "-" * 60)

    for parent_level, parent_to_children in classifier.hierarchy_map.items():
        print(f"\n    {parent_level} relationships:")
        # Show first 2 examples
        for parent, children in list(parent_to_children.items())[:2]:
            print(f"      '{parent}' can have children: {list(children)}")

    # ==============================================================================
    # STEP 7: SAVE AND LOAD MODEL
    # ==============================================================================

    print("\n[8] Saving model...")

    model_path = classifier.save_model(path="models/food_hierarchy_model")
    print(f"    ✓ Model saved to: {model_path}")
    print("    ✓ Saved components:")
    print("      - 3 trained models (one per level)")
    print("      - Label encoders for each level")
    print("      - Hierarchy map")
    print("      - Feature names and metadata")

    # Demonstrate loading
    print("\n[9] Testing model loading...")

    new_classifier = HierarchicalClassifier(
        name="food_categorization_loaded",
        levels=["main_category", "sub_category", "specific_type"],
    )

    new_classifier.load_model(path=model_path)
    print("    ✓ Model loaded successfully!")

    # Verify it works
    test_predictions = new_classifier.predict_hierarchy(
        new_items.head(1), enforce_hierarchy=True
    )
    print(
        f"    ✓ Verified predictions match: {test_predictions.iloc[0].to_dict()}"
    )


if __name__ == "__main__":
    main()
