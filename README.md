# 🤖 Neural Memory Networks for Question Answering

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated **End-to-End Memory Network** implementation for intelligent question answering, trained on the bAbI dataset from Facebook Research. This project demonstrates advanced neural network architectures capable of reading comprehension and logical reasoning.

## 🎯 Project Overview

This chatbot leverages **Memory Networks** to understand context, store relevant information, and answer questions based on given stories. Unlike traditional chatbots, this system can perform multi-hop reasoning over sequences of sentences to provide accurate answers.

### Key Features

- 🧠 **Memory Network Architecture**: Implements end-to-end memory networks for complex reasoning
- 📚 **Story Comprehension**: Processes narrative text and answers questions about it
- 🔄 **Multi-hop Reasoning**: Can connect multiple facts to derive answers
- 🎛️ **Attention Mechanism**: Uses soft attention to focus on relevant parts of the story
- 📊 **High Accuracy**: Achieves 91%+ accuracy on the bAbI dataset
- 🔧 **Customizable**: Easy to extend with new vocabulary and story types

## 🏗️ Architecture

The model uses a sophisticated architecture combining:

- **Input Encoder (m)**: Embeds story sentences into memory vectors
- **Input Encoder (c)**: Creates candidate memory representations  
- **Question Encoder**: Processes questions into query vectors
- **Attention Mechanism**: Computes relevance scores between queries and memories
- **Response Module**: Generates answers using LSTM and dense layers

```
Story + Question → Memory Networks → Attention → LSTM → Answer
```

## 📊 Dataset

Trained on the **bAbI Dataset** from Facebook Research:
- **Training Set**: 10,000 story-question-answer triplets
- **Test Set**: 1,000 examples
- **Vocabulary**: 37 unique words
- **Tasks**: Simple reasoning, location tracking, object manipulation

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow keras numpy matplotlib pickle
```

### Installation

```bash
git clone https://github.com/mustafasamy28/neural-memory-networks-qa.git
cd neural-memory-networks-qa
```

### Usage

1. **Load the pre-trained model:**
```python
from keras.models import load_model
model = load_model('chatbot_120_epochs.h5')
```

2. **Ask questions about stories:**
```python
# Your story (must use vocabulary from training set)
story = "John left the kitchen . Sandra dropped the football in the garden ."
question = "Is the football in the garden ?"

# Get prediction
answer = predict_answer(story, question)
print(f"Answer: {answer}")  # Output: "yes"
```

3. **Available vocabulary:**
```python
vocab = {'John', 'Mary', 'Sandra', 'Daniel', 'kitchen', 'garden', 
         'bedroom', 'bathroom', 'office', 'hallway', 'football', 
         'apple', 'milk', 'moved', 'went', 'got', 'took', 'dropped', 
         'left', 'grabbed', 'picked', 'put', 'discarded', 'travelled',
         'journeyed', 'back', 'there', 'down', 'up', 'to', 'in', 'the',
         'Is', 'yes', 'no', '.', '?'}
```

## 📈 Performance

- **Training Accuracy**: 96.15%
- **Validation Accuracy**: 91.20%
- **Training Time**: ~7 minutes (120 epochs)
- **Model Size**: Compact and efficient

### Training Curve
The model shows excellent convergence with minimal overfitting, achieving stable performance after ~80 epochs.

## 🔬 Technical Details

### Model Architecture
- **Embedding Dimensions**: 64
- **Memory Size**: Variable (up to 156 tokens)
- **LSTM Units**: 32
- **Dropout**: 0.3 (embeddings), 0.5 (final layer)
- **Optimizer**: RMSprop
- **Loss**: Categorical Crossentropy

### Key Components
1. **Vectorization**: Converts text to padded sequences
2. **Memory Attention**: Soft attention over story elements
3. **Response Generation**: LSTM-based answer synthesis
4. **End-to-end Training**: Joint optimization of all components

## 📁 Project Structure

```
Chatbot-Memoria-qa/
├── Chat-Bots.ipynb          # Main notebook with full implementation
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── data/
│   ├── train_qa.txt        # Training data (pickled)
│   └── test_qa.txt         # Test data (pickled)
├── models/
│   └── chatbot_120_epochs.h5  # Trained model weights
└── src/
    ├── data_preprocessing.py   # Data loading and preprocessing
    ├── model_architecture.py  # Network architecture
    └── utils.py               # Helper functions
```

## 🔧 Advanced Usage

### Custom Stories
Create your own stories using the available vocabulary:

```python
my_story = "Mary moved to the office . John got the apple ."
my_question = "Is Mary in the office ?"
answer = model_predict(my_story, my_question)
```

### Model Retraining
To retrain with custom data:

```python
# Prepare your data
custom_data = [(story, question, answer), ...]

# Vectorize and train
inputs, queries, answers = vectorize_stories(custom_data)
model.fit([inputs, queries], answers, epochs=50)
```

## 🎓 Research Background

This implementation is based on the groundbreaking paper:
> Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus.  
> **"End-To-End Memory Networks"**  
> *NIPS 2015* | [Paper](http://arxiv.org/abs/1503.08895)

The model demonstrates how neural networks can perform reasoning tasks that traditionally required symbolic AI approaches.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Support for larger vocabularies
- [ ] Multi-task learning across bAbI tasks
- [ ] Attention visualization
- [ ] Performance optimization
- [ ] Web interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

**Mostafa Samy**
- 📧 Email: mustafasamy28@gmail.com
- 🐙 GitHub: [@mustafasamy28](https://github.com/mustafasamy28)
- 💼 LinkedIn: [Mostafa Samy](https://www.linkedin.com/in/mostafa-samy-9b95711a7/)

## 🙏 Acknowledgments

- Facebook Research for the bAbI dataset
- The Memory Networks research community
- TensorFlow and Keras development teams

---

⭐ **Star this repository if you found it helpful!**
