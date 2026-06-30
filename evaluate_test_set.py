import os
import cv2
import torch
import numpy as np
from blueprint_ocr import CustomCRNN, greedy_ctc_decoder, load_vocab

def evaluate():
    model_weights = "best_model.pth"
    test_gt = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test\rec_gt.txt"
    test_dir = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test"
    
    # Load Vocab
    vocab_chars = load_vocab()
    idx2char = {idx: char for idx, char in enumerate(vocab_chars)}
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCRNN(num_classes=len(vocab_chars)).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    
    correct = 0
    total = 0
    
    print("--- Starting Evaluation ---")
    
    with open(test_gt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                img_rel_path = parts[0].strip()
                ground_truth = parts[1].strip()
                
                img_path = os.path.join(test_dir, img_rel_path)
                
                if not os.path.exists(img_path):
                    continue
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Preprocess matching the training loop
                resized = cv2.resize(img, (128, 32))
                norm_img = resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    logits = model(img_tensor)
                    decoded = greedy_ctc_decoder(logits.cpu(), idx2char)[0]
                
                total += 1
                if decoded == ground_truth:
                    correct += 1
                    try:
                        print(f"[CORRECT] File: {img_rel_path} | Pred: '{decoded}'")
                    except UnicodeEncodeError:
                        pass
                else:
                    try:
                        print(f"[WRONG] File: {img_rel_path} | Target: '{ground_truth}' | Pred: '{decoded}'")
                    except UnicodeEncodeError:
                        pass
                    
    print("\n--- Evaluation Results ---")
    print(f"Total Test Images: {total}")
    print(f"Correct Predictions (Rights): {correct}")
    print(f"Incorrect Predictions (Wrongs): {total - correct}")
    if total > 0:
        print(f"Accuracy: {(correct / total) * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
