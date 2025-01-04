# Speech_Enabled_Chatbot_Streamlit

A sophisticated chatbot application that combines speech recognition capabilities with TF-IDF-based natural language processing to provide intelligent responses. This application allows users to interact with the chatbot using both text and voice input through a clean Streamlit interface.

## 🌟 Features

- **Dual Input Methods**: 
  - Text-based input for typing questions
  - Speech recognition for voice-based interactions

- **Advanced NLP Processing**:
  - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
  - Cosine similarity for response matching
  - NLTK-based text preprocessing
  - Lemmatization and stop word removal

- **User-Friendly Interface**:
  - Clean Streamlit web interface
  - Persistent chat history
  - Real-time response generation
  - Easy switching between input methods

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/AchourTen/Speech_Enabled_Chatbot_Streamlit.git
cd Speech_Enabled_Chatbot_Streamlit
```


2. Make sure you have a working microphone for speech recognition features.

## 📦 Dependencies

- streamlit
- speech_recognition
- nltk
- scikit-learn
- string
- re


## 🚀 Usage

1. Prepare your training data:
   - Use/Create the text file (e.g., `F1.txt`) containing the knowledge base for your chatbot
   - Place it in the project directory

2. Run the application:
```bash
streamlit run chatbot.py
```

3. Using the chatbot:
   - Choose between text or speech input using the radio buttons
   - For text input: Type your question and click "Send"
   - For speech input: Click "Start Recording" and speak your question
   - View the chatbot's responses in the chat history

## 💡 How It Works

1. **Text Processing Pipeline**:
   - Input text is preprocessed (lowercase conversion, punctuation removal)
   - Words are tokenized and lemmatized
   - Stop words are removed
   - Text is vectorized using TF-IDF

2. **Response Generation**:
   - User input is processed through the same pipeline
   - Cosine similarity is calculated between the query and knowledge base
   - Most relevant response is selected based on similarity scores
   - Default responses are provided for low-confidence matches

3. **Speech Recognition**:
   - Audio input is captured through the microphone
   - Google Speech Recognition API converts speech to text
   - Resulting text is processed through the standard pipeline

## 🔍 Code Structure

```
speech-enabled-chatbot/
│
├── chatbot.py          # Main application file
├── F1.txt             # Knowledge base text file
└── README.md         # Documentation
```

## 🛠️ Configuration

You can modify these parameters in the code:

- Similarity threshold (default: 0.1)
- Speech recognition timeout (default: 5 seconds)
- Ambient noise adjustment duration (default: 1 second)


## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Streamlit team for the web interface framework
- Google Speech Recognition API for speech-to-text conversion

  
