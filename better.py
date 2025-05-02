import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define directories and parameters
train_dir = 'dataset/train'
val_dir = 'dataset/valid'
test_dir = 'dataset/test'
img_size = (224, 224)
batch_size = 32

# Load data with data augmentation during loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False
)

# Get class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

# Visualize some training images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

# Create a more robust data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

# Function to prepare dataset
def prepare_dataset(ds, augment=False):
    # Rescale pixel values
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                num_parallel_calls=AUTOTUNE)
    
    # Apply augmentation only to training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), 
                    num_parallel_calls=AUTOTUNE)
    
    # Use buffered prefetching
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare_dataset(train_ds, augment=True)
val_ds = prepare_dataset(val_ds)
test_ds = prepare_dataset(test_ds)

# Visualize augmented images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        # Convert back to 0-255 range for display
        plt.imshow(np.clip(images[i].numpy() * 255, 0, 255).astype("uint8"))
        plt.title(class_names[tf.argmax(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# Build model using transfer learning with EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Freeze the base model layers
base_model.trainable = False

# Build the model
model = models.Sequential([
    # The base model contains preprocessing, so we don't need separate rescaling
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model with learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks for early stopping and model checkpointing
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]



# Fine-tuning phase: unfreeze some of the base model layers
# Unfreeze the top layers of the model
base_model.trainable = True

# Freeze all the layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False
    
# Re-compile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

# Combine history objects
total_epochs = len(history.history['loss']) + len(fine_tune_history.history['loss'])
combined_acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
combined_val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
combined_loss = history.history['loss'] + fine_tune_history.history['loss']
combined_val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(total_epochs), combined_acc, label='Training Accuracy')
plt.plot(range(total_epochs), combined_val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(total_epochs), combined_loss, label='Training Loss')
plt.plot(range(total_epochs), combined_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.tight_layout()
plt.show()

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions on test data
test_images = []
test_labels = []
pred_labels = []

for images, labels in test_ds:
    predictions = model.predict(images)
    pred_labels.extend(np.argmax(predictions, axis=1))
    test_labels.extend(np.argmax(labels.numpy(), axis=1))
    test_images.extend(images.numpy())

test_images = np.array(test_images)
test_labels = np.array(test_labels)
pred_labels = np.array(pred_labels)

# Display some predictions
plt.figure(figsize=(15, 10))
for i in range(min(15, len(test_images))):
    plt.subplot(3, 5, i+1)
    plt.imshow(np.clip(test_images[i] * 255, 0, 255).astype('uint8'))
    
    true_class = class_names[test_labels[i]]
    pred_class = class_names[pred_labels[i]]
    
    color = 'green' if true_class == pred_class else 'red'
    plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Generate a confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(test_labels, pred_labels)

# Plot confusion matrix
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(test_labels, pred_labels, target_names=class_names))

# Save the model
model.save('sports_classification_model')
print("Model saved successfully!")