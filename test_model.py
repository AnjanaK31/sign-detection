import cv2
import torch
import os
import numpy as np
from blueprint_ocr import clean_direct_crop, CustomCRNN, greedy_ctc_decoder, load_vocab

if __name__ == "__main__":
    model_weights = "best_model.pth"
    test_image = r"D:\Cogentic\sign-detection\synthetic_generator\synthetic_data\images\syn_000003_1.png"
    
    # 1. Load Vocab
    vocab_chars = load_vocab()
    idx2char = {idx: char for idx, char in enumerate(vocab_chars)}
    
    # 2. Load Model
    model = CustomCRNN(num_classes=len(vocab_chars))
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    
    # 3. Load & Process Image
    crop_image = cv2.imread(test_image)
    
    # Try it WITHOUT the line cleaner first!
    # cleaned_crop = clean_direct_crop(crop_image)
    cleaned_crop = crop_image 
    
    gray = cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2GRAY) if len(cleaned_crop.shape) == 3 else cleaned_crop
    resized = cv2.resize(gray, (128, 32))
    
    # Save a debug image so you can see exactly what is being fed to the model
    cv2.imwrite("debug_network_input.jpg", resized)
    print("Saved what the network sees to: debug_network_input.jpg")
    
    norm_img = resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(img_tensor)
        decoded = greedy_ctc_decoder(logits, idx2char)
        
    print(f"Predicted Text: '{decoded[0]}'")
