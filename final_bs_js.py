import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision import models
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassF1Score
from transformers import ViTImageProcessor, ViTForImageClassification
from google.colab import drive
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from google.colab import drive
import zipfile
import os

drive.mount('/gdrive')

zip_path = '/gdrive/MyDrive/lung_image_sets.zip'

extract_path = '/gdrive/MyDrive/lung_image_sets'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

image_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

dataset = ImageFolder(root='lung_image_sets', transform=image_transforms)

label2id = dataset.class_to_idx
id2label = {v: k for k, v in label2id.items()}

train_dataset, test_dataset, val_dataset = random_split(dataset, [0.7, 0.2, 0.1])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=2, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=2, persistent_workers=True)

batch = next(iter(train_dataloader))
images, labels = batch
print(images.shape, labels.shape)

class ImageModel(L.LightningModule):
    def __init__(self, lr=1e-3, id2label={}, label2id={}):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=3)
        self.id2label = id2label
        self.label2id = label2id
        self.f1 = MulticlassF1Score(num_classes=3, average="macro")


    def step(self, batch, batch_idx, mode):
        x, y = batch
        output = self(x)
        output["loss"] = self.loss(output["logits"], y)
        output["class"] = torch.argmax(output["probs"], dim=1)
        output["accuracy"] = self.accuracy(output["class"], y)
        output["f1"] = self.f1(output["class"], y)
        output["labels"] = [self.id2label[i.item()] for i in output["class"]]
        if mode != "predict":
            self.log(f'{mode}_f1', output["f1"], prog_bar=True)
            self.log(f'{mode}_loss', output["loss"], prog_bar=True)
            self.log(f'{mode}_accuracy', output["accuracy"], prog_bar=True)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "predict")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class ViTModel(ImageModel):
    def __init__(self, lr=1e-4, id2label = id2label, label2id=label2id):
        super().__init__(lr, id2label=id2label, label2id=label2id)
        self.softmax = nn.Softmax(dim=1)
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', do_rescale=False)
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        )

    def forward(self, x):
        x = self.processor(images=x, return_tensors="pt")
        x = x.to(device)
        output = self.model(**x, output_attentions=True)
        logits = output.logits
        probs = self.softmax(logits)
        attentions = output.attentions
        return {"logits": logits, "probs": probs, "attentions": attentions}

class ConvolutionalModel(ImageModel):
    def __init__(self, lr=1e-4, id2label=id2label, label2id=label2id):
        super().__init__(lr, id2label=id2label, label2id=label2id)
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(3),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(3)
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(18432, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # ← dropout here
            nn.Linear(512, 10),
            nn.ReLU(),
            nn.Linear(10, len(id2label.keys()))
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convolution_stack(x)
        x = torch.flatten(x, 1)
        logits = self.linear_stack(x)
        probs = self.softmax(logits)
        return {"logits": logits, "probs": probs}

class ImagePreprocess(L.LightningModule):
    def __init__(self, size=224):
        super().__init__()
        self.transforms = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((size, size)),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def forward(self, x):
        return self.transforms(x)

class ResNetModel(ImageModel):
    def __init__(self, lr=1e-4, id2label=id2label, label2id=label2id):
        super().__init__(lr, id2label=id2label, label2id=label2id)
        self.base_model = models.resnet50(weights="IMAGENET1K_V1")
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, len(id2label))
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        probs = self.softmax(logits)
        return {"logits": logits, "probs": probs}


def compare_models_on_dataset(models, model_names, dataloader, id2label, num_classes=3):
    for model, name in zip(models, model_names):
        model.eval()
        model.to(device)

        accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
        f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

        all_probs = torch.zeros(num_classes).to(device)
        num_samples = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                preds = torch.argmax(output["probs"], dim=1)

                accuracy.update(preds, y)
                f1.update(preds, y)

                probs = output["probs"].sum(dim=0)  # sum over batch
                all_probs += probs
                num_samples += x.size(0)

        #normalize probabilities
        mean_probs = (all_probs / num_samples).detach().cpu().numpy()

        acc = accuracy.compute().item()
        f1_score_val = f1.compute().item()

        print(f"\n{name} Results on Test Set:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1_score_val:.4f}")

        #bar chart of average predicted probabilities
        plt.bar([id2label[i] for i in range(num_classes)], mean_probs)
        plt.title(f"{name} Average Prediction Confidence")
        plt.ylabel("Mean Probability")
        plt.ylim(0, 1)
        plt.savefig(f"/gdrive/MyDrive/mean_probability_{name}_comparison.png")  #save
        plt.show()


def show_vit_attention_map(image_pil, attentions):
    last_layer_attention = attentions[-1]  
    
    average_attention = last_layer_attention[0].mean(dim=0)  

    cls_token_attention = average_attention[0, 1:]

    grid_size = int(cls_token_attention.shape[0] ** 0.5)
    attention_map = cls_token_attention.reshape(grid_size, grid_size).cpu()

    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

    attention_map_resized = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)(
        attention_map.unsqueeze(0).unsqueeze(0)
    )[0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image_pil)
    axes[1].imshow(attention_map_resized, cmap='jet', alpha=0.5)
    axes[1].set_title("Attention Overlay (ViT)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("/gdrive/MyDrive/vit_attention_comparison.png")
    plt.show()

def analyze_vit_predictions(vit_model, dataloader, id2label, max_per_category=3, device="cuda"):
    vit_model.eval()
    vit_model.to(device)

    correct_shown, incorrect_shown = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = vit_model(x_batch)
            logits = output["logits"]
            probs = output["probs"]
            attentions = output["attentions"]
            preds = torch.argmax(probs, dim=1)

            for i in range(x_batch.size(0)):
                pred = preds[i].item()
                label = y_batch[i].item()
                image_tensor = x_batch[i].cpu()
                image_pil = to_pil_image(image_tensor)

                y_true.append(label)
                y_pred.append(pred)

                if pred == label and correct_shown < max_per_category:
                    print(f"Correctly classified as {id2label[pred]}")
                    show_vit_attention_map(image_pil, attentions)
                    correct_shown += 1

                elif pred != label and incorrect_shown < max_per_category:
                    print(f"Incorrectly classified: predicted {id2label[pred]}, actual {id2label[label]}")
                    show_vit_attention_map(image_pil, attentions)
                    incorrect_shown += 1

                if correct_shown >= max_per_category and incorrect_shown >= max_per_category:
                    break
            if correct_shown >= max_per_category and incorrect_shown >= max_per_category:
                break

   
def mc_dropout_predictions(model, x, num_samples=30):
  model.train()  #IMPORTANT: enables dropout
  model.to(device)

  preds = []
  with torch.no_grad():
      for _ in range(num_samples):
          out = model(x)
          preds.append(out["probs"].cpu())

  preds = torch.stack(preds)  
  mean_probs = preds.mean(dim=0)
  std_probs = preds.std(dim=0)

  return mean_probs, std_probs

def evaluate_standard(model, dataloader, name="Model", num_classes=3):
    model.eval()
    model.to(device)

    accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output["probs"], dim=1)

            accuracy.update(preds, y)
            f1.update(preds, y)

    acc = accuracy.compute().item()
    f1_val = f1.compute().item()

    print(f"\n{name} (Dropout OFF)")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1_val:.3f}")

def evaluate_mc_dropout(model, dataloader, name="Model (MC Dropout)", n_passes=30):
    model.eval()
    model.to(device)

    #dropout layers to remain active
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.apply(enable_dropout)

    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            #multiple stochastic forward passes
            probs_list = []
            for _ in range(n_passes):
                output = model(x)
                probs_list.append(output["probs"].unsqueeze(0))  

            probs_mc = torch.cat(probs_list, dim=0)           
            probs_mean = probs_mc.mean(dim=0)                
            preds = torch.argmax(probs_mean, dim=1)

            predictions.extend(preds.cpu().numpy())
            targets.extend(y.cpu().numpy())

    #metrics
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')

    print(f"\n{name} (Dropout ON via MC)")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")

def analyze_mc_uncertainty_dataset(model, dataloader, n_passes=30, id2label=None):
    model.eval()
    model.to(device)

    # Enable dropout layers in eval mode
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()
    model.apply(enable_dropout)

    stds = []
    means = []
    labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            probs_list = []
            for _ in range(n_passes):
                output = model(x)
                probs_list.append(output["probs"].cpu().numpy())  
            probs_mc = np.stack(probs_list) 
            mean_probs = probs_mc.mean(axis=0) 
            std_probs = probs_mc.std(axis=0)   

            #get the std dev of the top predicted class
            top_indices = np.argmax(mean_probs, axis=1)
            top_stds = std_probs[np.arange(len(top_indices)), top_indices]
            top_means = mean_probs[np.arange(len(top_indices)), top_indices]

            stds.extend(top_stds.tolist())
            means.extend(top_means.tolist())
            labels.extend(y.cpu().numpy())

    #histogram of std devs
    plt.figure(figsize=(6, 4))
    plt.hist(stds, bins=30, color='skyblue', edgecolor='black')
    plt.title("MC Dropout Prediction Uncertainty Across Dataset")
    plt.xlabel("Standard Deviation of Top Prediction")
    plt.ylabel("Number of Images")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    #scatter plot of confidence vs. uncertainty
    plt.figure(figsize=(6, 4))
    plt.scatter(means, stds, alpha=0.5, color='cornflowerblue')
    plt.title("Confidence vs. Uncertainty (MC Dropout)")
    plt.xlabel("Mean Confidence (Top Prediction)")
    plt.ylabel("Std Deviation (Uncertainty)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"Avg uncertainty (std dev): {np.mean(stds):.3f}")
    print(f"Most uncertain sample std: {np.max(stds):.3f}")

def plot_confusion_matrix(model, dataloader, id2label):
    #model is in evaluation mode
    model.eval()
    model.to(device)

    #true and predicted labels
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            output = model(images)
            preds = torch.argmax(output["logits"], dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    #display the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    display_labels = [id2label[i] for i in range(len(id2label))]

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(
        cmap="Blues", xticks_rotation=45
    )

    plt.title("Confusion Matrix (Full Test Set)")
    plt.tight_layout()
    plt.show()
    
def train(model):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,              #wait 3 epochs with no improvement
        mode='min',
        verbose=True
    )
    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    model_checkpoint = ModelCheckpoint("./checkpoints")

    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[early_stopping, model_checkpoint],
        logger=logger,
        val_check_interval=0.2,
        limit_val_batches=0.1
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path="last")
    return trainer, model_checkpoint

image = Image.open("/content/lung_image_sets/lung_aca/lungaca1000.jpeg").convert("RGB")
x = transforms.Resize((224, 224))(image)
x = transforms.ToTensor()(x).unsqueeze(0).to(device)

conv_model = ConvolutionalModel()
conv_trainer, conv_model_checkpoint = train(conv_model)

conv_model = ConvolutionalModel.load_from_checkpoint(conv_model_checkpoint.best_model_path)

conv_trainer.test(model=conv_model, dataloaders=test_dataloader)

conv_model.eval()
conv_model.to(device)

with torch.no_grad():
    conv_output = conv_model(x)

print("Conv:", conv_output["probs"])

mean, std = mc_dropout_predictions(conv_model, x)

labels = [id2label[i] for i in range(len(mean[0]))]
plt.bar(labels, mean[0], yerr=std[0])
plt.title("CNN Prediction with Uncertainty (MC Dropout)")
plt.ylabel("Confidence +/- Uncertainty")
plt.ylim(0, 1)
plt.savefig("/gdrive/MyDrive/convolution_mc_dropout.png")  #save to Drive
plt.show()

vit_model = ViTModel()
vit_model.train()
vit_trainer, vit_model_checkpoint = train(vit_model)

vit_model = ViTModel.load_from_checkpoint(vit_model_checkpoint.best_model_path)

vit_trainer.test(model=vit_model, dataloaders=test_dataloader)

vit_model.eval()
vit_model.cuda()

with torch.no_grad():
    vit_output = vit_model(x)
    
print("ViT:", vit_output["probs"])

analyze_vit_predictions(
    vit_model=vit_model,
    dataloader=test_dataloader,
    processor=vit_model.processor,
    id2label=id2label,
    max_per_category=10,  #show 10 correct and 10 incorrect examples
    device="cuda" if torch.cuda.is_available() else "cpu"
)

resnet_model = ResNetModel()
resnet_model.train()
resnet_trainer, resnet_ckpt = train(resnet_model)

resnet_model = ResNetModel.load_from_checkpoint(resnet_ckpt.best_model_path)
resnet_trainer.test(model=resnet_model, dataloaders=test_dataloader)

resnet_model.eval()
resnet_model.cuda()
with torch.no_grad():
    resnet_output = resnet_model(x)

print("ResNet:", resnet_output["probs"])

#dropout OFF 
evaluate_standard(conv_model, test_dataloader, name="CNN", num_classes=3)

#dropout ON
evaluate_mc_dropout(conv_model, test_dataloader, name="CNN", n_passes=30)

#evaluate with dropout OFF 
evaluate_standard(resnet_model, test_dataloader, name="ResNet", num_classes=3)

#evaluate with MC Dropout ON
evaluate_mc_dropout(resnet_model, test_dataloader, name="ResNet", n_passes=30)

analyze_mc_uncertainty_dataset(conv_model, test_dataloader, n_passes=30, id2label=id2label)

conv_model.eval()
conv_model.to(device)
resnet_model.eval()
resnet_model.to(device)
vit_model.eval()
vit_model.to(device)

print("Confusion Matrix for Conv Model")
plot_confusion_matrix(conv_model, test_dataloader, id2label)

print("Confusion Matrix for ResNet Model")
plot_confusion_matrix(resnet_model, test_dataloader, id2label)

print("Confusion Matrix for ViT Model")
plot_confusion_matrix(vit_model, test_dataloader, id2label)

