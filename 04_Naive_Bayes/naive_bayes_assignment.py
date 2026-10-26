from pathlib import Path
from collections import Counter
import random
import re
import math

# 1. SETTINGS
CATEGORIES = ["sci.space", "rec.sport.hockey"]

STOPWORDS = set("a an the and or but is are was were to of in on for with at by from as it this that be have has had i you he she we they not do does did".split())

# 2. LOAD DATA
data = []

for category in CATEGORIES:
    folder = Path("C:\\Users\\birav\\Desktop\\git\\UiA\\IKT466-1_26H\\04_Naive_Bayes\\20_newsgroups") / category

    for file in folder.iterdir():
        if file.is_file():
            text = file.read_text(encoding="latin-1")
            text = text.split("\n\n", 1)[-1]
            data.append((text, category))

random.Random(42).shuffle(data)

print("Total documents:", len(data))


# 3. PREPARE WORDS
def get_words(text, clean):
    if clean:
        text = text.lower()
        words = re.findall(r"[a-z]+", text)
        words = [word for word in words if word not in STOPWORDS]
    else:
        words = text.split()
    return words


# 4. TRAIN NAIVE BAYES
def train_model(training_data, clean):
    document_counts = Counter()
    word_counts = {category: Counter() for category in CATEGORIES}

    for text, category in training_data:
        document_counts[category] += 1
        word_counts[category].update(get_words(text, clean))

    vocabulary = set()
    for category in CATEGORIES:
        vocabulary.update(word_counts[category])

    return document_counts, word_counts, vocabulary


# 5. PREDICT THE CATEGORY
def predict(text, model, clean):
    document_counts, word_counts, vocabulary = model
    words = get_words(text, clean)
    scores = {}

    for category in CATEGORIES:
        # Prior: how common is this category?
        total_documents = sum(document_counts.values())
        score = math.log(document_counts[category] / total_documents)

        total_words = sum(word_counts[category].values())

        for word in words:
            if word not in vocabulary:
                continue

            # Laplace smoothing: +1 prevents zero probability
            probability = (word_counts[category][word] + 1) / (total_words + len(vocabulary))
            score += math.log(probability)

        scores[category] = score

    return max(scores, key=scores.get)


# 6. CALCULATE ACCURACY
def accuracy(dataset, model, clean):
    correct = 0

    for text, actual_category in dataset:
        predicted_category = predict(text, model, clean)
        if predicted_category == actual_category:
            correct += 1

    return correct / len(dataset)


# 7. RUN EXPERIMENTS
for train_ratio in [0.6, 0.7]:
    train_end = int(len(data) * train_ratio)
    test_start = int(len(data) * 0.8)

    training = data[:train_end]
    validation = data[train_end:test_start]
    testing = data[test_start:]

    for clean in [False, True]:
        model = train_model(training, clean)

        val_accuracy = accuracy(validation, model, clean)
        test_accuracy = accuracy(testing, model, clean)

        print(f"Training: {train_ratio:.0%} | Cleaned: {clean}")
        print(f"Validation accuracy: {val_accuracy:.3f}")
        print(f"Test accuracy:       {test_accuracy:.3f}")
        print()