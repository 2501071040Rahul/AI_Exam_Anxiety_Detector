from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

model.load_state_dict(torch.load("bert_anxiety_model.pt", map_location=torch.device('cpu')))
model.eval()

# Request format
class TextRequest(BaseModel):
    text: str

label_map = {
    0: "Low Anxiety",
    1: "Moderate Anxiety",
    2: "High Anxiety"
}

@app.post("/predict")
def predict_anxiety(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return {"anxiety_level": label_map[prediction]}
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

model.load_state_dict(torch.load("bert_anxiety_model.pt", map_location=torch.device("cpu")))
model.eval()


class TextRequest(BaseModel):
    text: str


label_map = {
    0: "Low Anxiety",
    1: "Moderate Anxiety",
    2: "High Anxiety"
}


@app.post("/predict")
def predict_anxiety(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return {"anxiety_level": label_map[prediction]}
@app.post("/predict")
def predict_anxiety(request: TextRequest):

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    label_map = {
        0: "Low Anxiety",
        1: "Moderate Anxiety",
        2: "High Anxiety"
    }

    return {"anxiety_level": label_map[prediction]}
