# College Chatbot and Homepage Clone
# College Chatbot

This repository contains the implementation of a retrieval-based chatbot designed for my college. The chatbot provides information about courses, fee structure, placements, and more. It leverages deep learning and NLP techniques to understand and respond to user queries effectively.

## Features

- **Deep Learning**: Powered by a neural network implemented using PyTorch.
- **NLP**: Utilizes tokenization, lemmatization, and stopword removal to process and understand queries.
- **Web Interface**: A user-friendly interface built with HTML, CSS, and JavaScript.
- **Data Generation**: Comprehensive datasets created for training the chatbot using various AI and LLM models.

## Technologies Used

- **Backend**: Python, PyTorch, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Data**: JSON for intents and responses

## Project Structure

- `static/`
  - `app.js`: JavaScript file for client-side logic.
  - `style.js`: CSS file for styling the web interface.
  - `images/`: Directory containing images used in the web interface.
- `templates/`
  - HTML files for the web interface.
- `train.py`: Script for training the chatbot model.
- `model.py`: Defines the neural network model for the chatbot.
- `nltk_utils.py`: Functions for tokenizing, lemmatizing, and processing text data.
- `chat.py`: Script for interacting with the trained chatbot.
- `app.py`: Flask application for serving the chatbot.
- `intents.json`: Contains the data for the chatbot, including user queries and responses.
- `requirements.txt`: List of dependencies required to run the project.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
