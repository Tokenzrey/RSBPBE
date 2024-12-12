from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
nlp = spacy.load("en_core_web_sm")

# Request model
class SentenceRequest(BaseModel):
    sentence: str

# Preprocessing function
def clean_text_bert_friendly(text):
    """Clean text while retaining BERT compatibility."""
    text = text.lower()
    text = re.sub(r'[^a-z\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization and embedding extraction
def get_bert_embeddings(sentence, tokenizer, model):
    tokens = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)

    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    tokens_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0].cpu())

    tokens_list = [token for token in tokens_list if token not in ["[CLS]", "[SEP]"]]
    embeddings = embeddings[1:len(tokens_list)+1]

    return tokens_list, embeddings

# Graph construction
def build_graph(tokens_list, embeddings, cleaned_sentence):
    doc = nlp(cleaned_sentence)
    G = nx.DiGraph()

    for idx, token in enumerate(tokens_list):
        G.add_node(idx, label=token, embedding=embeddings[idx])

    for token in doc:
        for child in token.children:
            weight = 1.0 / (abs(token.i - child.i) + 1)
            G.add_edge(token.i, child.i, weight=weight)

    similarity_matrix = cosine_similarity(embeddings)
    k = 2
    for i in range(len(tokens_list)):
        top_k_indices = np.argsort(similarity_matrix[i])[-k:]
        for j in top_k_indices:
            if i != j:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    for node in G.nodes():
        if node < len(similarity_matrix):
            avg_similarity = np.mean([similarity_matrix[node][j] for j in range(len(tokens_list)) if node != j])
            G.add_edge(node, node, weight=avg_similarity)

    return G

# Convert graph to JSON
def graph_to_json(G):
    graph_data = {
        "nodes": {
            int(n): {
                "label": G.nodes[n].get("label", ""),
                "embedding": G.nodes[n].get("embedding", np.zeros(768)).tolist()
            } for n in G.nodes()
        },
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "weight": float(G.edges[u, v].get("weight", 0.0))
            } for u, v in G.edges()
        ]
    }
    return graph_data

# Convert graph to PyTorch Geometric Data
def graph_to_data(graph):
    num_nodes = len(graph.nodes)
    feature_matrix = np.array([graph.nodes[node]['embedding'] for node in graph.nodes()])
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    edge_index = np.array(np.nonzero(adjacency_matrix))
    data = Data(
        x=torch.tensor(feature_matrix, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long)
    )
    return data

# Define GCN Model
class GCNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

@app.post("/build_graph/")
async def build_graph_endpoint(request: SentenceRequest):
    cleaned_sentence = clean_text_bert_friendly(request.sentence)
    tokens_list, embeddings = get_bert_embeddings(cleaned_sentence, tokenizer, model)
    G = build_graph(tokens_list, embeddings, cleaned_sentence)
    graph_json = graph_to_json(G)
    return graph_json

@app.post("/predict_sentiment/")
async def predict_sentiment(request: SentenceRequest):
    # Path to trained GCN model
    gcn_model_path = "gcn_model_cpu.pth"

    # Label map
    label_map = {0: "positive", 1: "neutral", 2: "negative"}

    # Load GCN model
    input_dim = 768
    hidden_dim = 64
    output_dim = len(label_map)
    gcn_model = GCNClassifier(input_dim, hidden_dim, output_dim)
    gcn_model.load_state_dict(torch.load(gcn_model_path))
    gcn_model.eval().to(device)

    # Preprocess text
    cleaned_text = clean_text_bert_friendly(request.sentence)

    # Get BERT embeddings
    tokens_list, embeddings = get_bert_embeddings(cleaned_text, tokenizer, model)

    # Build graph
    graph = build_graph(tokens_list, embeddings, cleaned_text)

    # Convert graph to PyTorch Geometric Data
    pyg_data = graph_to_data(graph)
    pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        out = gcn_model(pyg_data.x.to(device), pyg_data.edge_index.to(device), pyg_data.batch)
        pred = out.argmax(dim=1).item()

    return {"sentence": request.sentence, "predicted_sentiment": label_map[pred]}
