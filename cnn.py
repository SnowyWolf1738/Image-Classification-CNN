import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from PIL import Image
import shutil

# Caches images as 22x224 tensors, these transformations only need to be applied once per image instead of every single time the image is loaded
class CachedImageFolder(Dataset):
    def __init__(self, folder, cache_file, transform=None):
        self.transform = transform
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            ds = datasets.ImageFolder(folder)
            self.samples = []
            resize = transforms.Resize((224, 224))
            to_tensor = transforms.ToTensor()
            for path, label in ds.samples:
                img = Image.open(path).convert("RGB")
                img = resize(img)
                img = to_tensor(img)
                self.samples.append((img, label))
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Get Data Loader for images, using the cache
def get_loader(folder, cache_file, batch_size, shuffle,
               label_type="labeled", augmentation_level="heavy"):
    # Base cached dataset (already resized + toTensor)
    ds = CachedImageFolder(folder, cache_file)

    # Normalization parameters
    mean = [0.6002, 0.5747, 0.5495]
    std  = [0.2607, 0.2638, 0.2677]

    # Transform pipelines
    if augmentation_level == "heavy":
        tf = K.AugmentationSequential(
            K.RandomResizedCrop((224,224), scale=(0.7,1.0), same_on_batch=False),
            K.RandomHorizontalFlip(p=0.5,same_on_batch=False),
            K.RandomRotation(15.0,same_on_batch=False),
            K.ColorJitter(0.3,0.3,0.3,0.1,same_on_batch=False),
            K.RandomErasing(p=0.2, same_on_batch=False),
            K.Normalize(mean, std),
            data_keys=["input"],
        )
    elif augmentation_level == "light":
        tf = K.AugmentationSequential(
            K.RandomRotation(15, same_on_batch=False),
            K.RandomHorizontalFlip(p=0.5,same_on_batch=False),
            K.Normalize(mean, std),
            data_keys=["input"],
        )
    else:  # agumentation_level == "none" → validation
        tf = K.AugmentationSequential(
            K.Normalize(mean, std),
            data_keys=["input"],
        )

    # Collate function
    def collate_fn(batch):
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).cuda()

        # Apply transforms
        imgs = tf(imgs)

        if label_type == "labeled":
            labels = torch.tensor(labels).long().cuda()
        else:
            num_classes = len(set(labels))
            labels = torch.randint(0, num_classes, (len(labels),)).long().cuda()

        return imgs, labels

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn
    )

# Three Head CNN Model
class UnifiedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # IN-DOMAIN HEAD (ResNet18)
        self.in_head = models.resnet18(weights=None)
        self.in_head.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.in_head.fc.in_features, num_classes)
        )

        # DOMAIN CLASSIFIER HEAD 
        self.domain_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112x112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 56x56
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28x28
            nn.Flatten()
        )
        self.domain_fc = nn.Linear(128*28*28, 256)
        self.domain_head = nn.Linear(256, 2)  # in-domain vs out-domain

        # OUT-DOMAIN HEAD
        self.out_head = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )

    # FORWARD (returns all heads)
    def forward(self, x):
        # Domain head
        f = self.domain_features(x)
        f = self.domain_fc(f)
        domain_logits = self.domain_head(f)

        # In-domain head
        in_logits = self.in_head(x)

        # Out-domain head
        out_logits = self.out_head(x)

        return domain_logits, in_logits, out_logits

    # INFERENCE: automatic routing
    def classify(self, x):
        self.eval()
        with torch.no_grad():
            domain_logits = self.domain_head(self.domain_fc(self.domain_features(x)))
            dom = domain_logits.argmax(1)

            in_logits = self.in_head(x)
            out_logits = self.out_head(x)

            final = torch.zeros_like(in_logits)
            for i in range(x.size(0)):
                if dom[i] == 0:
                    final[i] = in_logits[i]
                else:
                    final[i] = out_logits[i]
            return final.softmax(1)

# Traing loop for domain classifier head
def train_domain_head(model, in_loader, out_loader, device, epochs=50, lr=0.01,
                      save_path="best_domain_head.pth"):
    model.to(device)
    model.domain_fc.train()
    model.domain_head.train()

    opt = torch.optim.Adam(
        list(model.domain_fc.parameters()) + list(model.domain_head.parameters()),
        lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0   # Track best domain accuracy

    for epoch in range(epochs):
        total, correct = 0, 0
        for (xin,_), (xout,_) in zip(in_loader, out_loader):
            xin = xin.to(device)
            xout = xout.to(device)

            # Labels: 0=in-domain, 1=out-domain
            x = torch.cat([xin, xout], dim=0)
            y = torch.cat([
                torch.zeros(xin.size(0)),
                torch.ones(xout.size(0))
            ]).long().to(device)

            opt.zero_grad()
            logits = model.domain_head(model.domain_fc(model.domain_features(x)))
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        acc = correct / total

        # Save best model (based on domain accuracy)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "domain_fc": model.domain_fc.state_dict(),
                "domain_head": model.domain_head.state_dict(),
            }, save_path)
    ckpt = torch.load(save_path, weights_only=True)
    model.domain_fc.load_state_dict(ckpt["domain_fc"])
    model.domain_head.load_state_dict(ckpt["domain_head"])

    # Delete the file after loading
    if os.path.exists(save_path):
        os.remove(save_path)

# Training loop for in-domain classifier head
def train_in_head(model, train_loader, device, epochs=110, lr=0.01):
    model.to(device)

    opt = torch.optim.SGD(model.in_head.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for m in model.in_head.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    for epoch in range(epochs):
        model.in_head.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model.in_head(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        sched.step()

# Training loop for out-domain classifier head
def train_out_head(model, out_loader, device, num_classes, epochs=5, lr=1e-3):
    model.to(device)
    model.out_head.train()

    opt = torch.optim.Adam(model.out_head.parameters(), lr=lr)
    target = torch.full((1, num_classes), 1/num_classes).to(device)

    for epoch in range(epochs):
        for x, _ in out_loader:
            x = x.to(device)

            opt.zero_grad()
            logits = model.out_head(x)
            probs = torch.softmax(logits, dim=1)

            # KL divergence to uniform distribution
            loss = F.kl_div(probs.log(), target.expand_as(probs), reduction="batchmean")
            loss.backward()
            opt.step()

def learn(path_to_in_domain, path_to_out_domain):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = "./cache"
    in_cache = "./cache/train.pkl"
    out_cache = "./cache/out.pkl"

    os.makedirs(cache_dir, exist_ok=True)

    in_loader = get_loader(path_to_in_domain, in_cache, batch_size=32,
                           shuffle=True, label_type="labeled", augmentation_level="heavy")
    
    domain_in_loader = get_loader(path_to_in_domain, in_cache, batch_size=32,
                                  shuffle=True, label_type="labeled", augmentation_level="light")
    
    domain_od_loader = get_loader(path_to_out_domain, out_cache, batch_size=32,
                                  shuffle=True, label_type="unlabeled", augmentation_level="light")

    num_classes = num_classes = len(os.listdir(path_to_in_domain))

    model = UnifiedModel(num_classes)

    train_domain_head(model, domain_in_loader, domain_od_loader, device, epochs=50)
    train_in_head(model, in_loader, device, epochs=110)
    train_out_head(model, domain_od_loader, device, num_classes, epochs=5)

    # Clear cache once training is finished
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    return model

def compute_accuracy(path_to_eval_folder, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size=32
    num_workers=0

    # Same preprocessing as used for the in-domain dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.6002, 0.5747, 0.5495], [0.2607, 0.2638, 0.2677])
    ])

    dataset = datasets.ImageFolder(path_to_eval_folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    model.eval()
    model.to(device)

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Model performs:
            # 1) domain classification
            # 2) automatic routing to in/out head
            # 3) output softmax probabilities
            preds = model.classify(images)
            pred_labels = preds.argmax(1)

            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    return correct / total