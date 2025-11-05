import streamlit as st
import torch
import torch.nn as nn

# --------------------------------------
# Simple example model architecture
# (replace with your actual model class)
# --------------------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)

# --------------------------------------
# Load all models
# --------------------------------------
@st.cache_resource
def load_model(path):
    model = DummyModel()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

models = {
    "Model 1": load_model("model1.pth"),
    "Model 2": load_model("model2.pth"),
    "Model 3": load_model("model3.pth"),
}

# --------------------------------------
# Streamlit UI
# --------------------------------------
st.title("🧠 PyTorch Model Inference Dashboard")
st.write("Select one of your saved models and run inference below.")

# Model selection
selected_model_name = st.selectbox("Select a model", list(models.keys()))
model = models[selected_model_name]

# Simple input
st.write("### Enter input values (comma-separated):")
user_input = st.text_input("Example: 0.5, 1.2, -0.7")

# Inference button
if st.button("Run Inference"):
    try:
        # Convert input string to tensor
        x = torch.tensor([list(map(float, user_input.split(',')))])
        with torch.no_grad():
            output = model(x)
        st.success(f"✅ Model Output: {output.numpy().flatten().tolist()}")
    except Exception as e:
        st.error(f"Error: {e}")
