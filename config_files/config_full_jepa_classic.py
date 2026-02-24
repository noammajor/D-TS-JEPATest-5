configJEPA = {
    # --- Global & Infrastructure ---
    "batch_size": 32,             # Standard batch size used in the paper 
    "path_save": "./output_model/JEPACLASSIC/",
    "checkpoint_save": 5000,
    
    # --- Data Loading Protocol ---
    "patch_size": 32,       
    "patch_size_forcasting": 32,            # Length of a single patch
    "ratpatchesio_": 10,          # Paper segments series into exactly 10 patches
    "patch_size_forecasting": 32,            # Length of a single patch
    "ratio_patches": 10,          # Paper segments series into exactly 10 patches 
    "mask_ratio": 0.7,            # Specific TS-JEPA ratio 
    "masking_type": "block",      # Using block masking as per your script
    "val_prec": 0.1,              # 10% validation split
    "test_prec": 0.25,            # 25% test split
    "val_prec_forcasting": 0.1,              # 10% validation split
    "test_prec_forcasting": 0.25,
    
    # --- Pre-training Optimization (JEPA) ---
    "lr": 1e-5,                   # Optimal LR for Weather/ETT datasets [cite: 199, 206]
    #"num_epochs": 5000,           # Total pre-training epochs
    "num_epochs": 1500,
    "ema_momentum": 0.998,        # Momentum m to prevent collapse [cite: 205]
    "ipe_scale": 1.25,
    "warmup_ratio": 0.15,
    "num_semantic_tokens": 8,     # Number of discrete tokens for JEPA
    
    # --- Architecture (Paper Standards) [cite: 194] ---
    "encoder_embed_dim": 128,      
    "encoder_nhead": 2,            
    "encoder_num_layers": 2,       
    "encoder_kernel_size": 3,
    "encoder_embed_bias": True,

    # --- Predictor Architecture [cite: 194] ---
    "predictor_embed": 128,
    "predictor_nhead": 2,
    "predictor_num_layers": 2,

    # --- Forecasting Downstream Task [cite: 82, 101] ---
    "epoch_t": 500,               # Epochs to train the frozen-encoder head
    "context_t": 32,              # Sequence context (patch size)
    "horizon_t": 3,              # Number of patches in the input sequence
    "patches_t": 10,              # Total patches to predict
    
    # --- Data Paths (All Paper Datasets) [cite: 92-95, 178-179] ---
    "data_paths": [
         './data/electricity.csv'
    ],
    "timestampcols": ['date'],
    # --- Input Variables for Each Dataset ---
    "input_variables_forcasting": [['OT']],
    # training will be univariate for forecasting as per the paper
    "input_variables": [
        # Electricity (Variables 0-319 + OT)
        [str(i) for i in range(320)] + ["OT"] 
    ],

}