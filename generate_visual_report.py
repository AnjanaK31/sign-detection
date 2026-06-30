import os
import cv2
import torch
import numpy as np
from blueprint_ocr import CustomCRNN, greedy_ctc_decoder, load_vocab

def generate_html_report():
    model_weights = "best_model.pth"
    test_gt = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test\rec_gt.txt"
    test_dir = r"d:\Cogentic\sign-detection\dataset_nived\dataset\test"
    output_html = "evaluation_report.html"
    
    # Load Vocab
    vocab_chars = load_vocab()
    idx2char = {idx: char for idx, char in enumerate(vocab_chars)}
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCRNN(num_classes=len(vocab_chars)).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    
    results = []
    correct = 0
    total = 0
    
    print("--- Running Evaluation for Visual Report ---")
    
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
                
                resized = cv2.resize(img, (128, 32))
                norm_img = resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(img_tensor)
                    decoded = greedy_ctc_decoder(logits.cpu(), idx2char)[0]
                
                is_correct = (decoded == ground_truth)
                if is_correct:
                    correct += 1
                total += 1
                
                # We need absolute path for the browser to render local images easily
                # Replacing backslashes with forward slashes for HTML
                abs_img_path = "file:///" + img_path.replace("\\", "/")
                results.append((abs_img_path, ground_truth, decoded, is_correct))
                
    # Sort results to show incorrect ones at the top for easy debugging
    results.sort(key=lambda x: x[3])
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("Generating HTML...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCR Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; }}
            .summary {{ display: flex; justify-content: space-around; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .summary-box {{ text-align: center; }}
            .summary-box h2 {{ margin: 0; font-size: 2em; }}
            .summary-box p {{ margin: 5px 0 0 0; color: #7f8c8d; }}
            table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .correct {{ color: #27ae60; font-weight: bold; }}
            .incorrect {{ color: #e74c3c; font-weight: bold; }}
            .img-container {{ max-height: 50px; }}
            img {{ height: 40px; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <h1>OCR Evaluation Report</h1>
        <div class="summary">
            <div class="summary-box"><h2>{total}</h2><p>Total Images</p></div>
            <div class="summary-box"><h2>{correct}</h2><p>Correct</p></div>
            <div class="summary-box"><h2>{total - correct}</h2><p>Incorrect</p></div>
            <div class="summary-box"><h2 style="color: {'#27ae60' if accuracy > 70 else '#e74c3c'};">{accuracy:.2f}%</h2><p>Accuracy</p></div>
        </div>
        
        <table>
            <tr>
                <th>Crop Image</th>
                <th>Ground Truth (Target)</th>
                <th>Model Prediction</th>
                <th>Status</th>
            </tr>
    """
    
    for img_path, target, pred, is_correct in results:
        status_text = "Correct" if is_correct else "Incorrect"
        status_class = "correct" if is_correct else "incorrect"
        
        html_content += f"""
            <tr>
                <td><img src="{img_path}" alt="crop" /></td>
                <td>{target}</td>
                <td>{pred}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
        """
        
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Done! Open '{output_html}' in your web browser to see the visual report.")

if __name__ == "__main__":
    generate_html_report()
