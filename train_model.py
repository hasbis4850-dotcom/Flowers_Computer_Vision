
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Settings
DATA_DIR = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 5

tf.random.set_seed(SEED)
np.random.seed(SEED)

# 2. Check dataset folders
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

for class_name in classes:
    folder = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    count = sum(
        1 for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"{class_name}: {count} images")

print("Dataset folders verified.")

# 3. Load dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_data.class_names

print("Flower classes:", class_names)

# 4. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train model
print("\nTraining started...")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# 6. Evaluate model
print("\nEvaluating model...")

y_true = []
y_pred = []

for images, labels in test_data:
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

# 7. Save confusion matrix
cm = confusion_matrix(
    y_true, y_pred, labels=list(range(len(class_names)))
)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
cm_display.plot(xticks_rotation=45)
plt.title("Flowers Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# 8. Save accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.close()

# 9. SAVE THE TRAINED MODEL
print("\nSaving trained model...")

model.save("flowers_model.keras")

with open("class_names.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(class_names))

print("Model saved as flowers_model.keras")
print("Class names saved as class_names.txt")
print("Training completed successfully.")

