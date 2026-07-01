import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time

# -----------------------------------------------------------------------------
# 0. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_vocab(dict_path=r"d:\Cogentic\sign-detection\synthetic_generator\dict.txt"):
    vocab_chars = ['blank']
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                c = line.rstrip('\n')
                if c and c not in vocab_chars:
                    vocab_chars.append(c)
    else:
        print(f"WARNING: Dictionary file not found at {dict_path}")
        # fallback
        vocab_chars.extend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                       'Ø', '°', 'M', 'X', '-', '.', 'I', 'S', 'O', ' ', 'A', 'V', '+', '±'])
    return vocab_chars

# -----------------------------------------------------------------------------
# 1. PREPROCESSING (LINE REMOVAL VIA EXPANSION)
# -----------------------------------------------------------------------------
def preprocess_and_clean_crop(full_image, obb_geometry, expansion=5):
    """
    Recovers a slightly expanded bounding box, identifies lines using Hough Transform,
    erases them via inpainting, and crops back to the exact target OBB frame.
    
    Args:
        full_image: Original full-resolution image (numpy array BGR or Grayscale).
        obb_geometry: 4 points of the bounding box [(x1,y1), (x2,y2), (x3,y3), (x4,y4)].
        expansion: Number of pixels to expand the bounding box for context.
    Returns:
        Cleaned, exactly cropped image region.
    """
    # Convert OBB to a format we can expand
    obb_pts = np.array(obb_geometry, dtype=np.float32)
    
    # Calculate bounding rectangle of the OBB
    rect = cv2.minAreaRect(obb_pts)
    (cx, cy), (w, h), angle = rect
    
    # Expand the bounding box
    exp_w, exp_h = w + 2 * expansion, h + 2 * expansion
    
    # Get perspective transform to warp the expanded region to a straight rectangle
    box_expanded = cv2.boxPoints(((cx, cy), (exp_w, exp_h), angle))
    
    # Target dimensions for expanded crop
    exp_w_int, exp_h_int = int(exp_w), int(exp_h)
    dst_pts_expanded = np.array([[0, exp_h_int],
                                 [0, 0],
                                 [exp_w_int, 0],
                                 [exp_w_int, exp_h_int]], dtype=np.float32)
    
    # Sort obb_pts to roughly match dst_pts (bottom-left, top-left, top-right, bottom-right)
    # This is a heuristic and might need adjustment based on exact OBB point ordering conventions
    # Assuming obb_pts are already somewhat ordered or we sort them by x/y
    
    M_expanded = cv2.getPerspectiveTransform(box_expanded.astype(np.float32), dst_pts_expanded)
    expanded_crop = cv2.warpPerspective(full_image, M_expanded, (exp_w_int, exp_h_int))
    
    # Grayscale and Inverse OTSU
    if len(expanded_crop.shape) == 3:
        gray = cv2.cvtColor(expanded_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = expanded_crop.copy()
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=50, minLineLength=w*0.5, maxLineGap=10)
    
    # Create mask for lines
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw line on mask (thickness can be adjusted based on typical line width)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
    # Inpaint to remove lines
    cleaned_expanded = cv2.inpaint(expanded_crop, line_mask, 3, cv2.INPAINT_TELEA)
    
    # Crop back to the original OBB size by removing the expansion borders
    h_cleaned, w_cleaned = cleaned_expanded.shape[:2]
    final_crop = cleaned_expanded[expansion:h_cleaned-expansion, expansion:w_cleaned-expansion]
    
    return final_crop

def clean_direct_crop(crop_image):
    """
    Directly cleans an already cropped image by removing straight lines 
    (like dimension underlines or strikethroughs) using Hough Transform and Inpainting.
    Perfect for when you already have the crops extracted by YOLOv8-OBB.
    """
    # Grayscale and Inverse OTSU
    if len(crop_image.shape) == 3:
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_image.copy()
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Hough Line Transform (tuned for smaller cropped regions)
    w = crop_image.shape[1]
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=30, minLineLength=max(15, w*0.3), maxLineGap=15)
    
    # Create mask for lines
    line_mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw line on mask. A thickness of 2 or 3 usually covers typical engineering lines.
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)
            
    # Inpaint to gracefully erase the lines while preserving overlapping text
    cleaned_crop = cv2.inpaint(crop_image, line_mask, 3, cv2.INPAINT_TELEA)
    
    return cleaned_crop

# -----------------------------------------------------------------------------
# 2. PYTORCH DATA PIPELINE
# -----------------------------------------------------------------------------
class BlueprintTextDataset(Dataset):
    def __init__(self, gt_filepath, base_dir, target_shape=(32, 128)):
        """
        Dataset for blueprint text parsing.
        Args:
            gt_filepath: Path to the ground truth text file.
            base_dir: Directory containing the crop_img folders.
            target_shape: (height, width) for resizing.
        """
        self.base_dir = base_dir
        self.target_shape = target_shape
        
        # Unique vocabulary based on requirements
        vocab_chars = load_vocab()
        
        self.char2idx = {char: idx for idx, char in enumerate(vocab_chars)}
        self.idx2char = {idx: char for idx, char in enumerate(vocab_chars)}
        self.vocab_size = len(vocab_chars)
        
        self.samples = []
        with open(gt_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Tab delimited as per requirements
                parts = line.split('\t')
                if len(parts) >= 2:
                    img_path = parts[0].strip()
                    text = parts[1].strip()
                    self.samples.append((img_path, text))

    def encode_text(self, text):
        # Ignore characters not in vocab to prevent errors, or you could map them to an <UNK> token
        encoded = [self.char2idx[c] for c in text if c in self.char2idx]
        return encoded

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, text = self.samples[idx]
        img_path = os.path.join(self.base_dir, img_rel_path)
        
        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for missing images
            img = np.zeros(self.target_shape, dtype=np.uint8)
            
        # Resize to fixed shape (32x128)
        img = cv2.resize(img, (self.target_shape[1], self.target_shape[0]))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension -> [1, 32, 128]
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        
        # Encode text
        encoded_text = self.encode_text(text)
        target = torch.tensor(encoded_text, dtype=torch.long)
        target_length = torch.tensor(len(encoded_text), dtype=torch.long)
        
        return img_tensor, target, target_length

def custom_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    # Concatenate targets into a 1D tensor for CTCLoss
    targets = torch.cat(targets, 0)
    target_lengths = torch.stack(target_lengths, 0)
    return images, targets, target_lengths

# -----------------------------------------------------------------------------
# 3. MODEL ARCHITECTURE (CRNN + CTC)
# -----------------------------------------------------------------------------
class CustomCRNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCRNN, self).__init__()
        
        # CNN Backbone
        # Input shape: [B, 1, 32, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> [B, 64, 16, 64]
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # -> [B, 128, 8, 32]
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # -> [B, 256, 4, 32]
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # -> [B, 256, 2, 32]
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))  # -> [B, 512, 1, 32]
        )
        
        # RNN Head
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True)
        
        # Linear Classifier
        # BiLSTM outputs 256 * 2 = 512 features
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [Batch, 1, Height, Width]
        conv_out = self.cnn(x)
        
        # Sequence Map Layer
        # conv_out is [Batch, 512, 1, Width_features]
        b, c, h, w = conv_out.size()
        assert h == 1, "Height must be 1 after CNN backbone"
        
        conv_out = conv_out.squeeze(2) # -> [Batch, 512, Width_features]
        # Permute to [Sequence_Length, Batch, Features] for LSTM
        conv_out = conv_out.permute(2, 0, 1) # -> [Width_features, Batch, 512]
        
        # RNN
        rnn_out, _ = self.rnn(conv_out) # -> [Width_features, Batch, 512]
        
        # Classifier
        logits = self.classifier(rnn_out) # -> [Width_features, Batch, num_classes]
        
        return logits

# -----------------------------------------------------------------------------
# 4. LOSS AND TRAINING JIG
# -----------------------------------------------------------------------------

def greedy_ctc_decoder(logits, idx2char, blank_idx=0):
    """
    Decodes the logits back into a string using greedy search.
    Args:
        logits: Tensor of shape [Seq_Len, Batch, Num_Classes]
    Returns:
        List of decoded strings
    """
    # Get the class with maximum probability
    # preds: [Seq_Len, Batch]
    preds = torch.argmax(logits, dim=2)
    
    # Transpose to [Batch, Seq_Len]
    preds = preds.permute(1, 0)
    
    decoded_strings = []
    for batch_idx in range(preds.size(0)):
        pred_seq = preds[batch_idx].tolist()
        decoded_chars = []
        prev_idx = -1
        
        for p in pred_seq:
            if p != blank_idx and p != prev_idx:
                decoded_chars.append(idx2char[p])
            prev_idx = p
            
        decoded_strings.append("".join(decoded_chars))
        
    return decoded_strings

def train_mini_jig(epochs=50, batch_size=8, lr=1e-4, save_path="best_model.pth"):
    print("Setting up training jig...")
    
    # Configure parameters
    train_gt = r"d:\Cogentic\sign-detection\dataset_nived\dataset\train\rec_gt.txt"
    train_dir = r"d:\Cogentic\sign-detection\dataset_nived\dataset\train"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Dataset and DataLoader
    dataset = BlueprintTextDataset(train_gt, train_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # Initialize Model, Loss, and Optimizer
    model = CustomCRNN(num_classes=dataset.vocab_size).to(device)
    
    # FIX: Load existing checkpoint to fine-tune rather than train from scratch
    if os.path.exists(save_path):
        print(f"Loading existing checkpoint from '{save_path}' for fine-tuning...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Checkpoint loaded. Fine-tuning with lr={lr}")
    else:
        print(f"No checkpoint found at '{save_path}', training from scratch with lr={lr}")
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    # Training Loop
    total_steps = epochs * len(dataloader)
    completed_steps = 0
    train_start = time.time()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for i, (images, targets, target_lengths) in enumerate(dataloader):
            step_start = time.time()

            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images) # [Seq_Len, Batch, Num_Classes]
            
            # Calculate input_lengths (Seq_Len for each sequence in the batch)
            seq_len = logits.size(0)
            batch_sz = logits.size(1)
            input_lengths = torch.full(size=(batch_sz,), fill_value=seq_len, dtype=torch.long, device=device)
            
            # CTC expects log_softmax
            log_probs = nn.functional.log_softmax(logits, dim=2)
            
            # Compute loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            completed_steps += 1

            if i % 10 == 0:
                elapsed       = time.time() - train_start
                avg_step_time = elapsed / completed_steps
                steps_left    = total_steps - completed_steps
                eta_secs      = avg_step_time * steps_left
                eta_h, rem    = divmod(int(eta_secs), 3600)
                eta_m, eta_s  = divmod(rem, 60)
                elapsed_h, er = divmod(int(elapsed), 3600)
                elapsed_m, elapsed_s = divmod(er, 60)

                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{i}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Elapsed: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:02d} | "
                    f"ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )
                
                # Show evaluation of the first batch using greedy decoder
                model.eval()
                with torch.no_grad():
                    sample_logits = model(images)
                    decoded_texts = greedy_ctc_decoder(sample_logits.cpu(), dataset.idx2char)
                    print("  Sample Predictions:")
                    for j in range(min(2, len(decoded_texts))):
                        safe_pred = decoded_texts[j].encode('ascii', errors='backslashreplace').decode('ascii')
                        print(f"    Pred: '{safe_pred}'")
                model.train()
                
        avg_loss   = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        ep_m, ep_s = divmod(int(epoch_time), 60)

        # Overall ETA after epoch
        elapsed      = time.time() - train_start
        epochs_left  = epochs - (epoch + 1)
        eta_secs     = (elapsed / (epoch + 1)) * epochs_left
        eta_h, rem   = divmod(int(eta_secs), 3600)
        eta_m, eta_s = divmod(rem, 60)

        print(
            f"\n{'='*70}\n"
            f"  Epoch {epoch+1}/{epochs} complete | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epoch Time: {ep_m:02d}m {ep_s:02d}s | "
            f"ETA for remaining epochs: {eta_h:02d}h {eta_m:02d}m {eta_s:02d}s"
            f"\n{'='*70}"
        )
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved new best model (loss {best_loss:.4f}) → {save_path}")

    total_time = time.time() - train_start
    tot_h, rem = divmod(int(total_time), 3600)
    tot_m, tot_s = divmod(rem, 60)
    print(f"\nTraining complete in {tot_h:02d}h {tot_m:02d}m {tot_s:02d}s. Best loss: {best_loss:.4f}")

def test_inference_on_crop(model_weights_path=None, test_image_path=None):
    """
    Helper function to run inference on a direct crop image.
    """
    # Initialize vocab
    vocab_chars = load_vocab()
    idx2char = {idx: char for idx, char in enumerate(vocab_chars)}
    
    model = CustomCRNN(num_classes=len(vocab_chars))
    if model_weights_path and os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    if test_image_path and os.path.exists(test_image_path):
        crop_image = cv2.imread(test_image_path)
        
        # 1. Clean the line from the crop
        cleaned_crop = clean_direct_crop(crop_image)
        
        # 2. Prepare for model
        gray = cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2GRAY) if len(cleaned_crop.shape) == 3 else cleaned_crop
        resized = cv2.resize(gray, (128, 32))
        norm_img = resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0) # [1, 1, 32, 128]
        
        # 3. Predict
        with torch.no_grad():
            logits = model(img_tensor)
            decoded = greedy_ctc_decoder(logits, idx2char)
            
        print(f"Predicted Text: '{decoded[0]}'")
        
        # Optionally show the before/after
        # cv2.imshow("Original Crop", crop_image)
        # cv2.imshow("Cleaned Crop", cleaned_crop)
        # cv2.waitKey(0)

if __name__ == "__main__":
    print("CRNN OCR Pipeline Initialized.")
    # Kick off training
    train_mini_jig(epochs=50, batch_size=8)
    
    # Example of how to run inference later:
    # test_inference_on_crop(model_weights_path="best_model.pth", test_image_path="path_to_your_crop.jpg")
