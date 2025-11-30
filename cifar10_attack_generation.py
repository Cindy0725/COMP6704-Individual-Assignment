import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import torchattacks  <-- Removed
import matplotlib.pyplot as plt
import pandas as pd
import time
import timm 
import os
import numpy as np

# ==========================================
# CUSTOM ATTACK IMPLEMENTATIONS
# ==========================================

class AttackBase:
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device

class FGSM(AttackBase):
    def __init__(self, model, eps=8/255, **kwargs):
        # FGSM is a one-step attack, ignores alpha/steps
        super().__init__(model, eps=eps)

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        adv_images = images + self.eps * data_grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()

class PGD(AttackBase):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, **kwargs):
        super().__init__(model, eps, alpha, steps)

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Random start (standard PGD practice)
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Projection
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
        return adv_images

class MIFGSM(AttackBase):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, **kwargs):
        super().__init__(model, eps, alpha, steps)
        self.decay = decay

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            
            # Normalize gradient (L1 norm)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = self.decay * momentum + grad
            
            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            
            # Projection
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
        return adv_images

class DIFGSM(AttackBase):
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.5, **kwargs):
        super().__init__(model, eps, alpha, steps)
        self.decay = decay
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_resize = torch.randint(low=img_resize, high=img_size, size=(1,)).item()
            rnd = img_resize
            h_rem = img_size - rnd
            w_rem = img_size - rnd
            pad_top = torch.randint(low=0, high=h_rem + 1, size=(1,)).item()
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem + 1, size=(1,)).item()
            pad_right = w_rem - pad_left
            
            padded = F.pad(F.interpolate(x, size=(rnd, rnd), mode='bilinear', align_corners=False),
                           (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            return padded if torch.rand(1) < self.diversity_prob else x
        return x

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Apply Input Diversity
            div_images = self.input_diversity(adv_images)
            
            outputs = self.model(div_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            grad = adv_images.grad.data
            
            # Normalize gradient
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = self.decay * momentum + grad
            
            adv_images = adv_images.detach() + self.alpha * momentum.sign()
            
            # Projection
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
        return adv_images

# ==========================================
# Configuration & Paths
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 
# EPOCHS = 5
EPOCHS = 10
MAX_STEPS = 10 
EPSILON = 8/255
ALPHA = 2/255

# Output Directory
SAVE_DIR = "/home/yzhangfg/benchmark_Qwen/COMP6704_Ass_to_be_deleted"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")
else:
    print(f"Saving to existing directory: {SAVE_DIR}")

# CIFAR-10 Class Names
CLASSES = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

print(f"Using device: {DEVICE}")

# ==========================================
# 1. Data Preparation
# ==========================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# ==========================================
# 2. Model Definitions
# ==========================================
def get_resnet():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

def get_vit():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
    return model.to(DEVICE)

# ==========================================
# 3. Training Function
# ==========================================
def train_model(model, name):
    filename = f"{name}_cifar10.pth"
    save_path = os.path.join(SAVE_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"Loading existing weights for {name} from {save_path}...")
        model.load_state_dict(torch.load(save_path))
        return model

    print(f"\nTraining {name}...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# ==========================================
# 4. Helper: Measure Time (Efficiency)
# ==========================================
def measure_efficiency(model, attack_class, loader, steps=None):
    model.eval()
    if steps:
        atk = attack_class(model, eps=EPSILON, alpha=ALPHA, steps=steps)
    else:
        atk = attack_class(model, eps=EPSILON)

    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    _ = atk(inputs, labels) # Warmup

    start_time = time.time()
    num_batches = 5
    for i, (inputs, labels) in enumerate(loader):
        if i >= num_batches: break
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        _ = atk(inputs, labels)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    return avg_time

# ==========================================
# 5. Helper: Convergence Analysis
# ==========================================
def get_convergence_data(model, attack_class, loader, max_steps):
    asr_history = []
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

    for s in range(1, max_steps + 1):
        atk = attack_class(model, eps=EPSILON, alpha=ALPHA, steps=s)
        adv_images = atk(inputs, labels)
        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        acc = (predicted.eq(labels).sum().item() / labels.size(0)) * 100
        asr_history.append(acc)
        
    return asr_history

# ==========================================
# 6. Helper: Visualization
# ==========================================
def visualize_all_attacks(model, loader):
    print("\nGenerating visual comparisons...")
    model.eval()
    
    # Get one image
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
    
    # We only need the first image in the batch
    img_tensor = inputs[0:1] 
    label_tensor = labels[0:1]
    true_label_name = CLASSES[label_tensor.item()]

    # Define all attacks
    attacks = {
        'Clean': None,
        'FGSM': FGSM(model, eps=EPSILON),
        'PGD': PGD(model, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'MIFGSM': MIFGSM(model, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'DIFGSM': DIFGSM(model, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS)
    }

    plt.figure(figsize=(16, 4))
    
    for i, (name, atk) in enumerate(attacks.items()):
        # Generate Image
        if atk is None:
            adv_img = img_tensor
        else:
            adv_img = atk(img_tensor, label_tensor)
        
        # Predict
        with torch.no_grad():
            output = model(adv_img)
            _, pred = output.max(1)
            pred_name = CLASSES[pred.item()]
        
        # Prepare for plotting (C,H,W -> H,W,C)
        # Clamp ensures values are valid for imshow [0,1]
        np_img = adv_img.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        
        # Plot
        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(np_img)
        
        # Color title red if attack succeeded (wrong prediction), green if failed (correct prediction)
        title_color = 'green' if pred_name == true_label_name else 'red'
        ax.set_title(f"{name}\nPred: {pred_name}", color=title_color, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"Original Class: {true_label_name}", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'attack_visual_comparison.png')
    plt.savefig(save_path)
    print(f"Saved visual comparison to {save_path}")

# ==========================================
# 7. Helper: Evaluate Accuracy & ASR
# ==========================================
def evaluate_attack_performance(model, attack_name, atk, loader, limit=200):
    """
    Returns: (Robust Accuracy, Attack Success Rate)
    """
    model.eval()
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        if total >= limit: break
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        if atk is None:
            # Clean evaluation
            adv_images = inputs
        else:
            # Attack generation
            adv_images = atk(inputs, labels)
            
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    robust_acc = 100 * correct / total
    # Attack Success Rate (ASR) is roughly (100 - Robust Accuracy)
    # technically ASR is calculated on images that were originally correctly classified, 
    # but (100 - Acc) is the standard proxy in simple benchmarks.
    asr = 100 - robust_acc 
    return robust_acc, asr

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # A. Load/Train Models
    resnet = get_resnet()
    resnet = train_model(resnet, "ResNet18")
    
    vit = get_vit()
    vit = train_model(vit, "ViT-Tiny")

    # B. Experiment 1: Efficiency
    print("\n=== Experiment 1: Computational Efficiency ===")
    iterative_attacks = {
        'PGD': PGD,
        'MIFGSM': MIFGSM,
        'DIFGSM': DIFGSM
    }
    time_results = {'Method': [], 'ResNet18 (s/batch)': [], 'ViT (s/batch)': []}
    
    # FGSM
    time_results['Method'].append('FGSM')
    time_results['ResNet18 (s/batch)'].append(measure_efficiency(resnet, FGSM, testloader))
    time_results['ViT (s/batch)'].append(measure_efficiency(vit, FGSM, testloader))

    # Iterative
    for name, atk_class in iterative_attacks.items():
        time_results['Method'].append(name)
        time_results['ResNet18 (s/batch)'].append(measure_efficiency(resnet, atk_class, testloader, steps=MAX_STEPS))
        time_results['ViT (s/batch)'].append(measure_efficiency(vit, atk_class, testloader, steps=MAX_STEPS))

    print(pd.DataFrame(time_results))
    
    # C. Experiment 2: Convergence Rate
    print("\n=== Experiment 2: Convergence Analysis ===")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, atk_class in iterative_attacks.items():
        acc_curve = get_convergence_data(resnet, atk_class, testloader, MAX_STEPS)
        plt.plot(range(1, MAX_STEPS+1), acc_curve, marker='o', label=name)
    plt.title("ResNet18: Robust Acc vs Iterations")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for name, atk_class in iterative_attacks.items():
        acc_curve = get_convergence_data(vit, atk_class, testloader, MAX_STEPS)
        plt.plot(range(1, MAX_STEPS+1), acc_curve, marker='o', label=name)
    plt.title("ViT: Robust Acc vs Iterations")
    plt.legend()
    plt.grid(True)
    
    plt_path = os.path.join(SAVE_DIR, 'convergence_plot.png')
    plt.savefig(plt_path)
    print(f"Saved convergence plot to {plt_path}")

    # D. Experiment 3: Transferability
    print("\n=== Experiment 3: Transferability ===")
    def evaluate_transfer(source_model, target_model, loader):
        # atk = PGD(source_model, eps=EPSILON, steps=10)
        atk = FGSM(source_model, eps=EPSILON)
        correct = 0; total = 0; limit = 200
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            adv_images = atk(inputs, labels)
            outputs = target_model(adv_images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            if total >= limit: break
        return 100 * correct / total

    print(f"ResNet -> ViT Acc: {evaluate_transfer(resnet, vit, testloader):.2f}%")
    print(f"ViT -> ResNet Acc: {evaluate_transfer(vit, resnet, testloader):.2f}%")

    # E. Experiment 4: Visual Comparison
    print("\n=== Experiment 4: Visual Comparison ===")
    visualize_all_attacks(resnet, testloader)

    # F. Experiment 5: Attack Performance Summary (Accuracy & ASR)
    print("\n=== Experiment 5: Attack Success Rate (ASR) Analysis ===")
    
    # Define attack instances for ResNet
    resnet_attacks = {
        'Clean': None,
        'FGSM': FGSM(resnet, eps=EPSILON),
        'PGD': PGD(resnet, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'MIFGSM': MIFGSM(resnet, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'DIFGSM': DIFGSM(resnet, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS)
    }

    # Define attack instances for ViT
    vit_attacks = {
        'Clean': None,
        'FGSM': FGSM(vit, eps=EPSILON),
        'PGD': PGD(vit, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'MIFGSM': MIFGSM(vit, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS),
        'DIFGSM': DIFGSM(vit, eps=EPSILON, alpha=ALPHA, steps=MAX_STEPS)
    }

    perf_data = {
        'Attack': [],
        'ResNet Acc (%)': [], 'ResNet ASR (%)': [],
        'ViT Acc (%)': [],    'ViT ASR (%)': []
    }

    for name in resnet_attacks.keys():
        # Evaluate ResNet
        r_acc, r_asr = evaluate_attack_performance(resnet, name, resnet_attacks[name], testloader)
        # Evaluate ViT
        v_acc, v_asr = evaluate_attack_performance(vit, name, vit_attacks[name], testloader)
        
        perf_data['Attack'].append(name)
        perf_data['ResNet Acc (%)'].append(f"{r_acc:.2f}")
        perf_data['ResNet ASR (%)'].append(f"{r_asr:.2f}")
        perf_data['ViT Acc (%)'].append(f"{v_acc:.2f}")
        perf_data['ViT ASR (%)'].append(f"{v_asr:.2f}")

    df_perf = pd.DataFrame(perf_data)
    print(df_perf)
    
    # Save CSV
    csv_path = os.path.join(SAVE_DIR, 'attack_performance_metrics.csv')
    df_perf.to_csv(csv_path, index=False)
    print(f"Saved performance metrics to {csv_path}")