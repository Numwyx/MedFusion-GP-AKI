import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import gpytorch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import optuna
from torch.cuda.amp import autocast, GradScaler

# === Random Seed ===
SEED = 1115
torch.manual_seed(SEED)

# === Output Directory ===
SAVE_PATH = "D:/data/123456789/"
os.makedirs(SAVE_PATH, exist_ok=True)

# === Model Components ===
class FeatureEmbedding(nn.Module):
    def __init__(self, cat_dims, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embed_dim) for n in cat_dims
        ])

    def forward(self, x_cat):
        embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embeds, dim=1)

class TabNetBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.attn = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x_proj = self.fc(x)
        attn_weights = self.attn(x_proj)
        return x * attn_weights

class SAINTBlock(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_attn, _ = self.attn(x, x, x)
        return self.norm(x + x_attn)

class ObliviousDecisionTreeLayer(nn.Module):
    def __init__(self, in_features, num_trees=8, tree_depth=3):
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.feature_selection = nn.Parameter(torch.randn(num_trees, tree_depth, in_features))
        self.thresholds = nn.Parameter(torch.randn(num_trees, tree_depth))
        self.leaf_values = nn.Parameter(torch.randn(num_trees, 2 ** tree_depth))

    def forward(self, x):
        batch_size = x.size(0)
        outputs = []
        for t in range(self.num_trees):
            feature_weights = torch.softmax(self.feature_selection[t], dim=-1)
            selected = torch.matmul(feature_weights, x.T).T
            decisions = torch.sigmoid(selected - self.thresholds[t]).unsqueeze(-1)
            complement = 1 - decisions
            probs = torch.stack([complement, decisions], dim=-1)
            paths = probs[:, 0, :, :]
            for d in range(1, self.tree_depth):
                paths = torch.bmm(
                    paths.view(batch_size, -1, 1),
                    probs[:, d, :, :].view(batch_size, 1, -1)
                ).view(batch_size, -1, 2)
            leaf_probs = paths.view(batch_size, -1)
            outputs.append(torch.matmul(leaf_probs, self.leaf_values[t]).unsqueeze(-1))
        return torch.cat(outputs, dim=-1)

class TabFusionNet(nn.Module):
    def __init__(self, cat_dims, embed_dim, cont_dim, num_trees=8, tree_depth=3):
        super().__init__()
        self.embed_layer = FeatureEmbedding(cat_dims, embed_dim)
        self.total_dim = embed_dim * len(cat_dims) + cont_dim
        self.tabnet = TabNetBlock(self.total_dim)
        self.saint = SAINTBlock(self.total_dim, heads=1)
        self.node = ObliviousDecisionTreeLayer(
            self.total_dim, num_trees=num_trees, tree_depth=tree_depth)
        self.out_layer = nn.Linear(num_trees, 1)

    def forward_features(self, x_cat, x_cont):
        x_embed = self.embed_layer(x_cat)
        x = torch.cat([x_embed, x_cont], dim=1)
        x = self.tabnet(x)
        x = x.unsqueeze(1)
        x = self.saint(x).squeeze(1)
        x = self.node(x)
        return self.out_layer(x)

class DKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

class TabFusionNetWithGP(nn.Module):
    def __init__(self, cat_dims, embed_dim, cont_dim, num_trees=8, tree_depth=3):
        super().__init__()
        self.backbone = TabFusionNet(cat_dims, embed_dim, cont_dim, num_trees, tree_depth)
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = None

    def forward(self, x_cat, x_cont):
        return self.backbone.forward_features(x_cat, x_cont)

    def init_gp(self, train_x, train_y):
        self.gp_model = DKLGPModel(train_x, train_y, self.gp_likelihood)

# === Data Loading & Preprocessing ===
print("Loading training set...")
train_df = pd.read_csv("D:/data/aki_noisy_10percent_final_corrected.csv")
print("Loading test set...")
test_df = pd.read_csv("D:/data/test1.csv")

cat_cols = ['TSCI-Level', 'VP-Use']
cont_cols = [c for c in train_df.columns if c not in cat_cols + ['AKI']]

# Label encoding with 'unknown' fallback
label_encoders = {}
for col in cat_cols:
    tmp = pd.concat([train_df[col].astype(str), pd.Series(['unknown'])], ignore_index=True)
    le = LabelEncoder().fit(tmp)
    label_encoders[col] = le

    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = test_df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'unknown')
    test_df[col] = le.transform(test_df[col])

# Standardize continuous features
scaler = StandardScaler().fit(train_df[cont_cols])

cat_dims = [len(label_encoders[c].classes_) for c in cat_cols]

# Convert to tensors
X_cat_train = torch.tensor(train_df[cat_cols].values, dtype=torch.long)
X_cont_train = torch.tensor(scaler.transform(train_df[cont_cols]), dtype=torch.float32)
y_train = torch.tensor(train_df['AKI'].values, dtype=torch.float32).view(-1, 1)

X_cat_test = torch.tensor(test_df[cat_cols].values, dtype=torch.long)
X_cont_test = torch.tensor(scaler.transform(test_df[cont_cols]), dtype=torch.float32)
y_test = torch.tensor(test_df['AKI'].values, dtype=torch.float32).view(-1, 1)

print(f"Training set: {len(train_df)}, Test set: {len(test_df)}")

# === Hyperparameter Optimization ===
def objective(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [4, 8, 16])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    num_trees = trial.suggest_int("num_trees", 4, 16)
    tree_depth = trial.suggest_int("tree_depth", 2, 4)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in kf.split(X_cat_train, y_train):
        Xc_tr, Xg_tr = X_cat_train[tr_idx], X_cont_train[tr_idx]
        Xc_va, Xg_va = X_cat_train[va_idx], X_cont_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model = TabFusionNetWithGP(cat_dims, embed_dim, Xg_tr.shape[1], num_trees, tree_depth).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler_cv = GradScaler()
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(30):
            model.train()
            optimizer.zero_grad()
            with autocast():
                out = model(Xc_tr.cuda(), Xg_tr.cuda())
                loss = loss_fn(out, y_tr.cuda())
            scaler_cv.scale(loss).backward()
            scaler_cv.step(optimizer)
            scaler_cv.update()

        model.eval()
        with torch.no_grad():
            logits = model(Xc_va.cuda(), Xg_va.cuda()).cpu().numpy()
            prob = 1 / (1 + np.exp(-logits))
            aucs.append(roc_auc_score(y_va.numpy(), prob))

    return float(np.mean(aucs))

print("Starting hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best parameters:", best_params)

# === Final Model Training ===
model = TabFusionNetWithGP(
    cat_dims, best_params['embed_dim'],
    X_cont_train.shape[1],
    best_params['num_trees'],
    best_params['tree_depth']
).to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
scaler_f = GradScaler()
loss_fn = nn.BCEWithLogitsLoss()

print("Training final model...")
for epoch in range(1, 501):
    model.train()
    optimizer.zero_grad()
    with autocast():
        out = model(X_cat_train.cuda(), X_cont_train.cuda())
        loss = loss_fn(out, y_train.cuda())
    scaler_f.scale(loss).backward()
    scaler_f.step(optimizer)
    scaler_f.update()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/500, Loss: {loss.item():.4f}")

# === Save Training Predictions ===
model.eval()
with torch.no_grad():
    logits = model(X_cat_train.cuda(), X_cont_train.cuda()).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    pd.DataFrame({
        "true": y_train.view(-1).numpy(),
        "prob": prob.squeeze()
    }).to_csv(os.path.join(SAVE_PATH, "train_results.csv"), index=False)

# === Test Evaluation ===
print("Evaluating on test set...")
with torch.no_grad():
    logits = model(X_cat_test.cuda(), X_cont_test.cuda()).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    test_auc = roc_auc_score(y_test.numpy(), prob)
    print(f"Test AUC: {test_auc:.4f}")
    pd.DataFrame({
        "true": y_test.view(-1).numpy(),
        "prob": prob.squeeze()
    }).to_csv(os.path.join(SAVE_PATH, "test_results.csv"), index=False)

# === 10-Fold Cross Validation (Training Set Only) ===
print("Starting 10-fold cross-validation...")
kf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
cv_tr, cv_va = [], []

for fold, (tr_idx, va_idx) in enumerate(kf10.split(X_cat_train, y_train), 1):
    print(f"Fold {fold}/10")
    Xc_tr, Xg_tr = X_cat_train[tr_idx], X_cont_train[tr_idx]
    Xc_va, Xg_va = X_cat_train[va_idx], X_cont_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    fold_model = TabFusionNetWithGP(
        cat_dims, best_params['embed_dim'],
        Xg_tr.shape[1],
        best_params['num_trees'],
        best_params['tree_depth']
    ).to("cuda")
    opt_fold = torch.optim.Adam(fold_model.parameters(), lr=best_params['lr'])
    scaler_cv = GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(100):
        fold_model.train()
        opt_fold.zero_grad()
        with autocast():
            out = fold_model(Xc_tr.cuda(), Xg_tr.cuda())
            loss = loss_fn(out, y_tr.cuda())
        scaler_cv.scale(loss).backward()
        scaler_cv.step(opt_fold)
        scaler_cv.update()

    fold_model.eval()
    with torch.no_grad():
        logits = fold_model(Xc_va.cuda(), Xg_va.cuda()).cpu().numpy()
        prob = 1 / (1 + np.exp(-logits))
        auc = roc_auc_score(y_va.numpy(), prob)
        print(f"Fold {fold} AUC: {auc:.4f}")

        tr_logits = fold_model(Xc_tr.cuda(), Xg_tr.cuda()).cpu().numpy()
        tr_prob = 1 / (1 + np.exp(-tr_logits))
        cv_tr.append(pd.DataFrame({
            "fold": fold - 1,
            "true": y_tr.view(-1).numpy(),
            "prob": tr_prob.squeeze()
        }))
        cv_va.append(pd.DataFrame({
            "fold": fold - 1,
            "true": y_va.view(-1).numpy(),
            "prob": prob.squeeze()
        }))

pd.concat(cv_tr).to_csv(os.path.join(SAVE_PATH, "cv_train_results.csv"), index=False)
pd.concat(cv_va).to_csv(os.path.join(SAVE_PATH, "cv_val_results.csv"), index=False)
print("Cross-validation results saved.")