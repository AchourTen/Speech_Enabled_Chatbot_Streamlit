import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SpeechEnabledChatbot:
    def __init__(self, filepath):
        """Initialize the chatbot with the given text file and speech recognition."""
        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Load and preprocess the text file
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split into sentences and preprocess
        self.sentences = sent_tokenize(text)
        self.processed_sentences = [self.preprocess(sentence) for sentence in self.sentences]
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_sentences)
    
    def preprocess(self, text):
        """Preprocess the text by removing punctuation, converting to lowercase,
        removing stop words, and lemmatizing."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Tokenize into words
        words = word_tokenize(text)
        
        # Remove stop words and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and not word.isnumeric()]
        
        return " ".join(words)
    
    def get_most_relevant_sentence(self, query):
        """Find the most relevant sentence in the corpus for the given query."""
        # Preprocess the query
        processed_query = self.preprocess(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity between query and all sentences
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        
        # Get index of most similar sentence
        most_similar_idx = similarities[0].argmax()
        return self.sentences[most_similar_idx], similarities[0][most_similar_idx]
    
    def get_response(self, query):
        """Generate a response to the user's query."""
        # Get most relevant sentence
        relevant_sentence, similarity = self.get_most_relevant_sentence(query)
        
        # If similarity is too low, return a default response
        if similarity < 0.1:
            return "I'm not sure about that. Could you please rephrase your question?"
        
        return relevant_sentence
    
    def transcribe_speech(self):
        """Record and transcribe speech input"""
        try:
            with sr.Microphone() as source:
                st.write("Listening... Speak now!")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5)
                st.write("Processing speech...")
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    return "Sorry, I couldn't understand what you said."
                except sr.RequestError:
                    return "Sorry, there was an error with the speech recognition service."
        except Exception as e:
            return f"Error accessing microphone: {str(e)}"

def main():
    st.title("Speech-Enabled Chatbot")
    
    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create chatbot instance (only once)
    if 'chatbot' not in st.session_state:
        # Replace 'F1.txt' with the path to your text file
        st.session_state.chatbot = SpeechEnabledChatbot('/Users/aymen/Desktop/gomycode/python/speech/chatbot/F1.txt')
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Text", "Speech"])
    
    if input_method == "Text":
        # Text input
        user_input = st.text_input("Ask a question:")
        if st.button("Send") and user_input:
            response = st.session_state.chatbot.get_response(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))
    
    else:  # Speech input
        if st.button("Start Recording"):
            text = st.session_state.chatbot.transcribe_speech()
            
            if text and not text.startswith(("Sorry", "Error")):
                response = st.session_state.chatbot.get_response(text)
                st.session_state.chat_history.append(("You", text))
                st.session_state.chat_history.append(("Bot", response))
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"👤 **You:** {message}")
        else:
            st.write(f"🤖 **Bot:** {message}")

if __name__ == "__main__":
    main()