# Sequence-to-Sequence-LSTM-Model-for-Language-Translation

This project implements a **Sequence-to-Sequence (Seq2Seq) Encoder–Decoder** model using **LSTM networks in TensorFlow/Keras** to translate English sentences into French. The system is trained on a 10,000-sentence bilingual dataset and includes custom preprocessing, tokenization, model building, inference decoding, and BLEU-based evaluation.

## Key Features

- LSTM-based Encoder–Decoder architecture  
- Custom preprocessing, tokenization, and vocabulary generation  
- Padded sequence creation for encoder/decoder  
- Greedy decoding inference model  
- BLEU score evaluation on test samples  
- Experiments with latent dimensions: 128, 256, 512  

## Dataset

The model uses the **fra.txt** bilingual English–French dataset.  
Only the **last 10,000 sentence pairs** are used for training and evaluation.

## Text Preprocessing

- Unicode normalization to strip accents  
- Regex cleaning (lowercasing, punctuation spacing)  
- Tokenization using Keras `Tokenizer`  
- `<start>` and `<end>` tokens added to French sentences  
- Padding to max sequence length

## Model Architecture

### Encoder
- Input: padded English sequences  
- Embedding layer  
- LSTM returning hidden + cell states  

### Decoder
- Input: French sequences with `<start>` token  
- Embedding layer  
- LSTM initialized with encoder states  
- Dense softmax layer generating tokens  

### Loss & Optimization
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `RMSprop`  

## Training

The model trains for:
- **10 epochs**
- **Batch size: 64**
- **80/20 train-test split**
- Additional **20% validation split** during training  

## Evaluation: BLEU Score

BLEU score is computed on a subset (200 samples) of test data using greedy decoding.

Experiments conducted with:
- latent_dim = 128  
- latent_dim = 256  
- latent_dim = 512  

(Insert BLEU scores printed from output here.)

## Effect of Sequence Length on Performance

Sequence length significantly reduces translation quality in vanilla Seq2Seq models:

- Encoder must compress full sentence into one vector → information loss  
- Decoder error accumulates across long outputs  
- BLEU-4 penalizes mistakes heavily on long sequences  
- Larger LSTMs (256/512 units) help but do **not** solve the core limitation  

**Conclusion:**  
A vanilla Seq2Seq LSTM without attention performs poorly on long sequences.  
Attention mechanisms dramatically improve translation accuracy.

## Future Work

- Add Bahdanau/Luong Attention  
- Use bidirectional encoder  
- Replace greedy decoding with beam search  
- Train on larger dataset  
- Use Transformer-based architecture  

## How to Run

Install dependencies:
```
pip install tensorflow numpy nltk
```

Run:
```
python seq2seq_translation.py
```

## Project Structure

```
│── seq2seq_translation.py
│── fra.txt
│── README.md
│── results/
│     ├── bleu_scores.txt
│     └── example_translations.txt
```
