import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from nsclc_dataset import NsclcImageDataset, get_nsclc_transforms
from nsclc_image_surv_model import NsclcImageSurvModel
from survival_head import cox_ph_loss


def train_nsclc_image_encoder(
    manifest_csv: str,
    batch_size: int = 2,
    num_epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cuda",
):

    ds = NsclcImageDataset(manifest_csv, transforms=get_nsclc_transforms())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    model = NsclcImageSurvModel(emb_dim=512).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            img = batch["image"].to(device)  # [B, 2, D, H, W]
            times = batch["time"].to(device)  # [B]
            events = batch["event"].to(device)  # [B]

            opt.zero_grad()
            risk, _ = model(img)
            loss = cox_ph_loss(risk, times, events)

            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            n_batches += 1

        print(
            f"Epoch {epoch+1}/{num_epochs}  loss={running_loss / max(n_batches,1):.4f}"
        )

    return model
