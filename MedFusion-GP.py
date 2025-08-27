import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import gpytorch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import optuna
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import warnings
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.means import LinearMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.distributions import MultivariateNormal

# Suppress GPyTorch warnings and configure device
warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
SEED = 1115
torch.manual_seed(SEED)

# Configure save path
SAVE_PATH = Path("D:/data/hyperfusion_test/")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# === Data Preparation ===
print("Loading and preprocessing data...")
df = pd.read_csv("D:/data/aki_balanced_smotenc.csv")

# Identify categorical and continuous columns
cat_cols = ['TSCI-Level', 'VP-Use']
cont_cols = [col for col in df.columns if col not in cat_cols + ['AKI']]

# Preprocess categorical columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Extract features and target
X_cat_all = df[cat_cols].values
X_cont_all = StandardScaler().fit_transform(df[cont_cols])
y_all = df['AKI'].values

# Split data into train/test sets
Xc_train_np, Xc_test_np, Xg_train_np, Xg_test_np, y_train_np, y_test_np = train_test_split(
    X_cat_all, X_cont_all, y_all, test_size=0.3, random_state=SEED, stratify=y_all
)

# Convert to PyTorch tensors and move to device
Xc_train = torch.tensor(Xc_train_np, dtype=torch.long).to(device)
Xc_test = torch.tensor(Xc_test_np, dtype=torch.long).to(device)
Xg_train = torch.tensor(Xg_train_np, dtype=torch.float32).to(device)
Xg_test = torch.tensor(Xg_test_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1).to(device)

# Calculate total training samples for ELBO
N_TRAIN = len(y_train_np)

# === Model Architecture Components ===

class HierarchicalFeatureEmbedding(nn.Module):
    """Hierarchical embedding layer for categorical features with attention mechanism"""
    def __init__(self, cat_dims, embed_dim):
        """
        Args:
            cat_dims: List of dimensions for each categorical feature
            embed_dim: Embedding dimension size
        """
        super().__init__()
        # Create embedding layers for each categorical feature
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(dim, embed_dim, padding_idx=0) for dim in cat_dims
        ])
        # Attention mechanism to capture feature interactions
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        # Layer normalization for stabilized outputs
        self.norm = nn.LayerNorm(embed_dim * len(cat_dims))

    def forward(self, x_cat):
        """Forward pass for hierarchical embeddings"""
        # Generate embeddings for each categorical feature
        embeds = [layer(x_cat[:, i]) for i, layer in enumerate(self.embedding_layers)]
        
        # Stack embeddings and apply attention
        stacked = torch.stack(embeds, dim=1)
        attn_out, _ = self.attention(stacked, stacked, stacked)
        
        # Create enhanced embeddings with residual connections
        enhanced = [e + 0.2 * attn_out[:, i, :] for i, e in enumerate(embeds)]
        
        # Concatenate and normalize final embeddings
        return self.norm(torch.cat(enhanced, dim=1))


class MultiModalFeatureFusion(nn.Module):
    """Feature fusion module for combining categorical and continuous features"""
    def __init__(self, cat_dim, cont_dim):
        """
        Args:
            cat_dim: Dimension of categorical features
            cont_dim: Dimension of continuous features
        """
        super().__init__()
        # Categorical feature processing tower
        self.cat_tower = nn.Sequential(
            nn.Linear(cat_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256)
        )
        
        # Continuous feature processing tower
        self.cont_tower = nn.Sequential(
            nn.Linear(cont_dim, 512), 
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256), 
            nn.InstanceNorm1d(256)
        )
        
        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        
        # Post-fusion processing
        self.post_fusion = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.SiLU(), 
            nn.Dropout(0.3), 
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512)
        )

    def forward(self, x_cat, x_cont):
        """Forward pass for feature fusion"""
        # Process each feature type separately
        c = self.cat_tower(x_cat)
        d = self.cont_tower(x_cont)
        
        # Combine features and generate fusion gate
        comb = torch.cat([c, d], dim=1)
        gate = self.fusion_gate(comb)
        
        # Apply gated fusion
        gated = torch.cat([c * gate[:, :256], d * gate[:, 256:]], dim=1)
        
        # Process fused features with residual connection
        return self.post_fusion(gated) + 0.1 * comb


class MegaResidualTransformer(nn.Module):
    """Transformer block with residual connections and pyramidal compression"""
    def __init__(self, dim, num_layers=8, heads=8):
        """
        Args:
            dim: Input dimension
            num_layers: Number of transformer layers
            heads: Number of attention heads
        """
        super().__init__()
        # Stack transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=heads, 
                dim_feedforward=4*dim, 
                dropout=0.1, 
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Learnable residual weights
        self.res_weights = nn.Parameter(torch.ones(num_layers))
        
        # Feature compression pyramid
        self.pyramid = nn.Sequential(
            nn.Linear(dim, dim//2), 
            nn.ReLU(), 
            nn.Linear(dim//2, dim//4), 
            nn.GELU()
        )

    def forward(self, x):
        """Forward pass through transformer block"""
        # Add sequence dimension
        x_ = x.unsqueeze(1)
        res = x_
        
        # Process through transformer layers with residual connections
        for i, layer in enumerate(self.layers):
            out = layer(x_)
            x_ = self.res_weights[i]*out + (1-self.res_weights[i])*res
            res = x_
        
        # Flatten and compress features
        flat = x_.squeeze(1)
        comp = self.pyramid(flat)
        
        # Return concatenated features
        return torch.cat([flat, comp], dim=1)


class ObliviousDecisionTreeLayer(nn.Module):
    """Differentiable decision tree layer with attention mechanism"""
    def __init__(self, in_features, num_trees=16, tree_depth=4):
        """
        Args:
            in_features: Input feature dimension
            num_trees: Number of trees in the layer
            tree_depth: Depth of each tree
        """
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        
        # Feature selection networks for each tree depth
        self.feature_selection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 128), 
                nn.ReLU(), 
                nn.Linear(128, in_features)
            for _ in range(tree_depth)
        ])
        
        # Tree parameters
        self.thresholds = nn.Parameter(torch.randn(num_trees, tree_depth))
        self.leaf_values = nn.Parameter(torch.randn(num_trees, 2**tree_depth))
        
        # Attention mechanism configuration
        attn_dim = max(4, (num_trees//4)*4)
        self.tree_attn = nn.MultiheadAttention(
            embed_dim=attn_dim, 
            num_heads=4, 
            batch_first=True
        )
        self.projection = nn.Linear(num_trees, attn_dim)
        
        # Output adapter (identity if dimensions match)
        self.output_adapter = nn.Sequential(
            nn.Linear(attn_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, num_trees)
        ) if attn_dim != num_trees else nn.Identity()

    def forward(self, x):
        """Forward pass through differentiable decision trees"""
        B = x.size(0)  # Batch size
        all_out = []
        
        # Process each tree independently
        for t in range(self.num_trees):
            decisions = []
            # Make decisions at each tree level
            for d in range(self.tree_depth):
                # Feature selection and threshold comparison
                w = torch.softmax(self.feature_selection[d](x), dim=-1)
                sel = torch.sum(w * x, dim=1, keepdim=True)
                decisions.append(torch.sigmoid(sel - self.thresholds[t,d]))
            
            # Calculate tree paths
            path = decisions[0]
            for d in decisions[1:]:
                path = torch.cat([path * d, path * (1-d)], dim=-1)
            
            # Handle path length (pad or truncate)
            Np = 2**self.tree_depth
            if path.size(-1) < Np:
                pad = torch.zeros(B, Np-path.size(-1), device=x.device)
                path = torch.cat([path, pad], dim=1)
            elif path.size(-1) > Np:
                path = path[:, :Np]
            
            # Calculate tree output
            lv = self.leaf_values[t].unsqueeze(0).expand(B, -1)
            out = torch.sum(path * lv, dim=1, keepdim=True)
            all_out.append(out)
        
        # Combine tree outputs
        trees = torch.cat(all_out, dim=1)
        
        # Apply attention to tree outputs
        proj = self.projection(trees)
        attn_out, _ = self.tree_attn(proj.unsqueeze(1), proj.unsqueeze(1), proj.unsqueeze(1))
        
        return self.output_adapter(attn_out.mean(dim=1))


class NeuralDeepForest(nn.Module):
    """Deep forest composed of multiple differentiable tree layers"""
    def __init__(self, input_dim, num_trees=32, tree_depth=6, forest_layers=3):
        """
        Args:
            input_dim: Input feature dimension
            num_trees: Number of trees per layer
            tree_depth: Depth of each tree
            forest_layers: Number of forest layers
        """
        super().__init__()
        layers = []
        dim = input_dim
        
        # Create forest layers
        for _ in range(forest_layers):
            layers.append(ObliviousDecisionTreeLayer(dim, num_trees, tree_depth))
            dim = num_trees  # Output dimension becomes input for next layer
        
        self.forest_layers = nn.ModuleList(layers)

    def forward(self, x):
        """Forward pass through deep forest"""
        feats = []
        for layer in self.forest_layers:
            x = layer(x)
            feats.append(x)  # Collect outputs from all layers
        
        # Concatenate outputs from all forest layers
        return torch.cat(feats, dim=1)


class DKLGaussianProcess(ApproximateGP):
    """Deep Kernel Learning Gaussian Process"""
    def __init__(self, inducing_points, input_dim):
        """
        Args:
            inducing_points: Initial inducing points
            input_dim: Dimension of input features
        """
        # Configure variational distribution
        variational_dist = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_dist,
            learn_inducing_locations=True, 
            jitter_val=1e-2
        )
        super().__init__(variational_strategy)
        
        # Configure mean and covariance modules
        self.mean_module = LinearMean(input_size=input_dim)
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=input_dim))

    def forward(self, x_proj):
        """Forward pass through GP layer"""
        return MultivariateNormal(
            self.mean_module(x_proj),
            self.covar_module(x_proj)
        )


class HyperFusionNet(nn.Module):
    """HyperFusion Network: Integrates embeddings, fusion, transformer, forest and GP"""
    def __init__(self, cat_dims, cont_dim, embed_dim=32, 
                 num_trees=32, tree_depth=6, num_layers=8):
        """
        Args:
            cat_dims: Dimensions of categorical features
            cont_dim: Dimension of continuous features
            embed_dim: Embedding dimension
            num_trees: Number of trees in forest
            tree_depth: Depth of each tree
            num_layers: Number of transformer layers
        """
        super().__init__()
        # Feature embedding module
        self.embedding = HierarchicalFeatureEmbedding(cat_dims, embed_dim)
        
        # Feature fusion module
        self.fusion = MultiModalFeatureFusion(len(cat_dims)*embed_dim, cont_dim)
        
        # Feature processing tower
        self.feature_tower = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(), 
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024), 
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.2)
        )
        
        # Transformer block
        self.mega_transformer = MegaResidualTransformer(
            1024, 
            num_layers=num_layers, 
            heads=8
        )
        
        # Deep forest module
        trans_out_dim = 1024 + 256
        self.deep_forest = NeuralDeepForest(
            trans_out_dim, 
            num_trees, 
            tree_depth, 
            forest_layers=3
        )
        folk_dim = 3 * num_trees  # Output dimension from deep forest
        
        # Feature extractor for DKL
        self.dkl_feature = nn.Sequential(
            nn.Linear(folk_dim, 512), 
            nn.ReLU(), 
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Linear(256, 64)  # Project to 64D for GP
        )
        
        # Initialize inducing points
        self.inducing_points = nn.Parameter(torch.randn(100, 64) * 0.1)
        
        # Gaussian Process layer
        self.gp = DKLGaussianProcess(self.inducing_points, input_dim=64)
        
        # Auxiliary output layer
        self.aux_output = nn.Sequential(
            nn.Linear(folk_dim, 512), 
            nn.ReLU(), 
            nn.Linear(512, 1)
        )

    def forward(self, x_cat, x_cont):
        """Forward pass through entire network"""
        # Process categorical features
        x_emb = self.embedding(x_cat)
        
        # Fuse categorical and continuous features
        fused = self.fusion(x_emb, x_cont)
        
        # Process through feature tower
        tower = self.feature_tower(fused)
        
        # Process through transformer
        trans = self.mega_transformer(tower)
        
        # Process through deep forest
        forest_out = self.deep_forest(trans)
        
        # Handle numerical instabilities
        forest_out = torch.nan_to_num(forest_out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Generate auxiliary predictions
        aux = self.aux_output(forest_out).squeeze(-1)
        
        # Extract deep features for GP
        feat64 = self.dkl_feature(forest_out)
        
        # Pass through Gaussian Process
        gp_out = self.gp(feat64)
        
        return aux, gp_out


class HyperFusionSystem:
    """End-to-end system for training and inference"""
    def __init__(self, cat_dims, cont_dim, device, num_train, **kwargs):
        """
        Args:
            cat_dims: Dimensions of categorical features
            cont_dim: Dimension of continuous features
            device: Computation device (CPU/GPU)
            num_train: Number of training samples (for ELBO)
            kwargs: Hyperparameters for HyperFusionNet
        """
        self.device = device
        self.num_train = num_train
        
        # Initialize model
        self.model = HyperFusionNet(cat_dims, cont_dim, **kwargs).to(self.device)
        
        # Initialize likelihood
        self.likelihood = BernoulliLikelihood().to(self.device)
        
        # Combine parameters
        params = list(self.model.parameters()) + list(self.likelihood.parameters())
        
        # Configure optimizer
        self.optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
        
        # Configure learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Binary cross entropy loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, x_cat, x_cont, y):
        """Single training step"""
        self.model.train()
        self.likelihood.train()
        self.optimizer.zero_grad()
        
        with autocast():  # Mixed precision context
            # Forward pass
            aux_pred, gp_out = self.model(x_cat, x_cont)
            y_flat = y.squeeze(-1)
            
            # Calculate losses
            loss_bce = self.bce_loss(aux_pred, y_flat)
            elbo = VariationalELBO(self.likelihood, self.model.gp, num_data=self.num_train)
            loss_gp = -elbo(gp_out, y_flat)
            total = loss_bce + loss_gp
        
        # Backpropagation with gradient scaling
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total.item(), loss_bce.item(), loss_gp.item()

    def predict(self, x_cat, x_cont):
        """Generate predictions"""
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), autocast():
            # Generate predictions
            aux_pred, gp_out = self.model(x_cat, x_cont)
            
            # Process GP output
            p_gp = self.likelihood(gp_out).mean.squeeze()
            
            # Process auxiliary output
            p_aux = torch.sigmoid(aux_pred)
            
            # Combine predictions
            p_final = (p_aux + p_gp) / 2
        
        return p_final.cpu().numpy()


# === Hyperparameter Optimization with Optuna ===
def objective(trial):
    """Objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 64])
    num_trees = trial.suggest_int('num_trees', 16, 64, step=16)
    tree_depth = trial.suggest_int('tree_depth', 4, 8, step=2)
    num_layers = trial.suggest_int('num_layers', 4, 12, step=4)
    
    # Configure cross-validation
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(Xg_train_np, y_train_np)):
        print(f"  Fold {fold+1}/3", end=' | ')
        
        # Prepare fold data
        Xc_tr, Xc_val = Xc_train_np[train_idx], Xc_train_np[val_idx]
        Xg_tr, Xg_val = Xg_train_np[train_idx], Xg_train_np[val_idx]
        y_tr, y_val = y_train_np[train_idx], y_train_np[val_idx]
        
        # Convert to tensors
        Xc_tr_t = torch.tensor(Xc_tr, dtype=torch.long).to(device)
        Xg_tr_t = torch.tensor(Xg_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1).to(device)
        Xc_val_t = torch.tensor(Xc_val, dtype=torch.long).to(device)
        Xg_val_t = torch.tensor(Xg_val, dtype=torch.float32).to(device)
        
        # Initialize system with current hyperparameters
        system = HyperFusionSystem(
            cat_dims=[len(np.unique(Xc_train_np[:,0]))+1, len(np.unique(Xc_train_np[:,1]))+1],
            cont_dim=Xg_train.shape[1],
            device=device,
            num_train=len(train_idx),
            embed_dim=embed_dim,
            num_trees=num_trees,
            tree_depth=tree_depth,
            num_layers=num_layers
        )
        
        # Train for 50 epochs
        for epoch in range(1, 51):
            system.train_step(Xc_tr_t, Xg_tr_t, y_tr_t)
        
        # Generate predictions and calculate AUC
        preds = system.predict(Xc_val_t, Xg_val_t)
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f"AUC: {auc:.4f}")
        
        # Clean up memory
        del system, Xc_tr_t, Xg_tr_t, y_tr_t, Xc_val_t, Xg_val_t
        torch.cuda.empty_cache()
    
    # Return mean AUC across folds
    return np.mean(aucs)


print("Starting hyperparameter search...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5, timeout=12*3600)  # 12-hour timeout
params = study.best_params
print("Best params:", params)


# === Final Model Training ===
print("\nTraining final model with best hyperparameters...")
final_system = HyperFusionSystem(
    cat_dims=[len(np.unique(Xc_train_np[:,0]))+1, len(np.unique(Xc_train_np[:,1]))+1],
    cont_dim=Xg_train.shape[1],
    device=device,
    num_train=N_TRAIN,
    **params
)

# Train for 100 epochs
for epoch in range(1, 101):
    total, bce_l, gp_l = final_system.train_step(Xc_train, Xg_train, y_train)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss:{total:.4f} (BCE:{bce_l:.4f}, GP:{gp_l:.4f})")


# === Prediction and Results Saving ===
def build_df(y_t, probs):
    """Build results dataframe with predictions and uncertainty"""
    y_np = y_t.cpu().numpy().squeeze()
    u = 1 - np.abs(probs - 0.5)*2  # Uncertainty measure
    return pd.DataFrame({'true': y_np, 'prob': probs, 'uncertainty': u})

# Generate and save results
print("\nSaving results...")
train_df = build_df(y_train, final_system.predict(Xc_train, Xg_train))
test_df  = build_df(y_test, final_system.predict(Xc_test, Xg_test))

train_df.to_csv(SAVE_PATH / "split_train_results.csv", index=False)
test_df.to_csv(SAVE_PATH / "split_test_results.csv", index=False)

print(f"Done. Results saved to: {SAVE_PATH}")
print(f"Final Train AUC: {roc_auc_score(train_df['true'], train_df['prob']):.4f}")
print(f"Final Test AUC:  {roc_auc_score(test_df['true'], test_df['prob']):.4f}")