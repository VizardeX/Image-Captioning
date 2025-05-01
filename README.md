# Automatic Image Captioning with Deep Learning

This project builds a deep learning system that generates natural language **captions for images** by combining **CNN-based visual feature extraction** and **RNN-based language modeling**.

---

## Overview

- Extracts image features using **InceptionV3** (pretrained on ImageNet).
- Generates captions using a **CNN + LSTM** model.
- Trained and validated on a structured version of the **Flickr8k** dataset.
- Generates captions for test images and exports results to a CSV file.

---

## Dataset Access

The dataset used for this project (`Assignment 2 files.zip`) is hosted on google drive due to size limits.

🔗 [Download from Google Drive](https://drive.google.com/file/d/16UyrY6BQQ_m9_l71EGrUejn-NqjZfooY/view?usp=sharing)

> Make sure to keep the folder zipped, since the first cell of the notebook will unzip it.


---

## 📁 Dataset Structure
```
Assignment 2 files/
├── train/         # Training images
├── val/           # Validation images
├── test/          # Test images (no captions)
├── train.txt      # Training image-caption pairs
├── val.txt        # Validation image-caption pairs
```


---

## Workflow

1. **Preprocessing**
   - Tokenizes and sequences captions.
   - Adds `<start>` and `<end>` tokens.
   - Computes `max_length` and vocabulary size.

2. **Feature Extraction**
   - Uses InceptionV3 to extract 2048-d image features.
   - Features are cached using `.pkl` files for faster reuse.

3. **Model Architecture**
   - Dense layer for image features.
   - Embedding + LSTM for caption input.
   - Combined output passed through Dense layers to predict the next word.

4. **Training**
   - Uses `sparse_categorical_crossentropy` as the loss function.
   - Includes **EarlyStopping** (`patience=15`) to prevent overfitting.
   - Monitors both training and validation loss.

5. **Caption Generation**
   - Custom `generate_caption()` function builds captions word-by-word.
   - Results are saved to `submission.csv` in the format: image_id,caption --> 123456.jpg,A man is riding a bike.



---

## Tools Used

- Python 3
- TensorFlow / Keras
- InceptionV3
- NumPy, pandas

---
  
> Built using TensorFlow and Google Colab.

