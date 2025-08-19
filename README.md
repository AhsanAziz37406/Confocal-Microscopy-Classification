🧪 Confocal Microscopy Image Classification

This repository presents a comprehensive framework for classifying confocal microscopy images using deep learning, transformer-based models, and hybrid DL + ML approaches. The work follows a structured workflow including preprocessing, feature extraction, multiple classification pipelines, and extensive evaluation.

🔄 Preprocessing

Confocal microscopy images undergo systematic preprocessing before being fed into learning models:

Data loading: Images are organized in class-wise folders and loaded into MATLAB.

Resizing and normalization: Each image is resized to match the input dimensions required by CNN and transformer architectures, and pixel values are normalized to stabilize training.

Augmentation: To enhance model robustness and mitigate overfitting, data augmentation strategies such as rotation, flipping, translation, and scaling are applied.

Dataset splitting: The dataset is divided into training, validation, and testing sets to ensure unbiased evaluation.

🔍 Feature Extraction

Two major strategies are employed for feature extraction:

CNN-based feature extraction: Pretrained CNNs (DarkNet53, DenseNet201, ResNet, etc.) are fine-tuned and used to extract deep features from the last pooling or fully connected layers.

Transformer-based embeddings: Vision transformer (ViT) and its variants are applied to capture spatial dependencies and long-range contextual information in confocal images.

🧠 Classification Pipelines

Three different experimental pipelines are implemented and compared:

1. End-to-End CNN Training

Pretrained CNNs are fine-tuned directly on the confocal dataset.

The final softmax layer is adapted to the dataset classes.

This approach benefits from hierarchical spatial feature extraction inherent to CNNs.

2. DL Feature Extraction + ML Classifiers

Deep features are extracted from CNN or transformer encoders.

These features are then fed into traditional ML classifiers such as SVM, KNN, Decision Trees, and Ensemble methods.

This hybrid pipeline allows benchmarking deep feature quality independent of end-to-end CNN training.

3. Transformer Architectures and Variants

Vision Transformers (ViT) and other transformer variants are trained/fine-tuned on the dataset.

Different architectural configurations are tested to capture long-range dependencies and self-attention maps.

Comparative evaluation of transformer models against CNNs highlights the strengths and limitations of attention-based learning for confocal image classification.

📊 Evaluation

To validate model performance and generalizability, a multi-level evaluation is carried out:

Metrics: Accuracy, precision, recall, and F1-score are calculated for all pipelines.

Confusion matrices: Used to visualize class-level performance and identify systematic errors.

Grad-CAM analysis: Class activation maps are generated for CNNs to explain decision-making before and after augmentation.

Cross-pipeline comparison: Results are compared across end-to-end CNNs, hybrid DL+ML, and transformer variants to determine the most effective strategy.

External validation: Additional unseen confocal images are tested to evaluate model robustness.
