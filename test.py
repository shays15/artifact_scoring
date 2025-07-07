import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils import ConvBlock2d, SimpleEncoder, ContrastiveMRIDataset

# ==== CONFIG ====
MODEL_PATH = "models/best_model.pt"
TEST_DATA_DIR = "data"          # path to test dir
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
model = SimpleEncoder(in_ch=1, out_ch=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== LOAD DATA ====
test_dataset = ContrastiveMRIDataset(TEST_DATA_DIR, 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ==== INFERENCE ====
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data["input"].to(DEVICE)
        outputs = model(inputs)

        # Save outputs
        torch.save(outputs.cpu(), output_dir / f"output_{i:03d}.txt")

print("Inference complete. Results saved to:", output_dir)
