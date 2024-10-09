import markdown
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup

# Load Pretrained Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Flask App - SearchRight
app = Flask(__name__, static_folder='static', template_folder='templates')

# Function to Parse Markdown File
def parse_markdown(file_content):
    text = file_content.decode("utf-8")
    blocks = re.split(r'\n{2,}', text)  # Split by two or more newline characters to create blocks
    return blocks

# Generate Embeddings for Blocks
def generate_embeddings(blocks):
    embeddings = model.encode(blocks)
    return embeddings

# Find Similar Blocks Based on Cosine Similarity
def find_similar_blocks(clicked_block, blocks, embeddings, top_n=3):
    clicked_embedding = model.encode([clicked_block])
    similarity_scores = cosine_similarity(clicked_embedding, embeddings)[0]
    similar_indices = np.argsort(similarity_scores)[-top_n-1:-1][::-1]  # Get top_n similar blocks
    return [(blocks[i], similarity_scores[i]) for i in similar_indices]

# Optional: Search Wikipedia for Similar Content
def search_wikipedia(query):
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return [p.get_text() for p in paragraphs[:3]]  # Return top 3 paragraphs

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    blocks = parse_markdown(file.read())
    embeddings = generate_embeddings(blocks)
    return render_template('index.html', blocks=blocks, embeddings=embeddings.tolist())

@app.route('/find_similar', methods=['POST'])
def find_similar():
    clicked_block = request.form['clicked_block']
    blocks = request.form.getlist('blocks')
    embeddings = np.array(request.form.getlist('embeddings')).astype(np.float32)
    similar_blocks = find_similar_blocks(clicked_block, blocks, embeddings)
    return render_template('index.html', blocks=blocks, similar_blocks=similar_blocks, embeddings=embeddings.tolist())

@app.route('/search_wikipedia', methods=['POST'])
def wikipedia_search():
    query = request.form['query']
    wikipedia_content = search_wikipedia(query)
    return render_template('index.html', wikipedia_content=wikipedia_content)

# Main Entry Point
if __name__ == '__main__':
    app.run(debug=True)
