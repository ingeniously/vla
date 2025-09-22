import argparse
import cv2
import numpy as np
import os
import torch
import vila_u
from vila_u.utils.tokenizer import tokenize_conversation
from vila_u.constants import IMAGE_TOKEN_INDEX

def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].detach().permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)

def direct_generate_image(model, prompt, cfg=3.0, batch_size=1):
    """
    Generate images directly by running LLM once, then RQ-Transformer, then RQ-VAE
    Bypasses problematic code paths in the model's generation methods
    """
    device = next(model.parameters()).device
    
    # 1. Prepare conditional and unconditional prompts
    conversation = [{"from": "human", "value": prompt}]
    cond_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True, image_generation=True).to(device)
    
    cfg_conversation = [{"from": "human", "value": " "}]
    uncond_ids = tokenize_conversation(cfg_conversation, model.tokenizer, add_generation_prompt=True, image_generation=True).to(device)
    
    # 2. Get the number of image tokens needed
    n_positions = model.vision_tower.image_tokens  # Usually 256 for 16×16 images
    
    # 3. Append image token placeholders
    cond_with_ph = torch.cat([cond_ids, torch.full((n_positions,), IMAGE_TOKEN_INDEX, dtype=cond_ids.dtype, device=device)])
    uncond_with_ph = torch.cat([uncond_ids, torch.full((n_positions,), IMAGE_TOKEN_INDEX, dtype=uncond_ids.dtype, device=device)])
    
    # 4. Build batch with proper padding (handle different lengths)
    max_len = max(cond_with_ph.size(0), uncond_with_ph.size(0))
    pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else 0
    
    # Pad sequences to the same length
    cond_padded = torch.full((max_len,), pad_id, dtype=cond_with_ph.dtype, device=device)
    uncond_padded = torch.full((max_len,), pad_id, dtype=uncond_with_ph.dtype, device=device)
    
    cond_padded[:cond_with_ph.size(0)] = cond_with_ph
    uncond_padded[:uncond_with_ph.size(0)] = uncond_with_ph
    
    # Create attention masks (1 where we have tokens, 0 for padding)
    cond_mask = torch.zeros((max_len,), dtype=torch.long, device=device)
    uncond_mask = torch.zeros((max_len,), dtype=torch.long, device=device)
    
    cond_mask[:cond_with_ph.size(0)] = 1
    uncond_mask[:uncond_with_ph.size(0)] = 1
    
    # Stack into batches
    input_ids = torch.stack([cond_padded, uncond_padded])
    attention_mask = torch.stack([cond_mask, uncond_mask])
    
    # 5. Replace placeholder tokens with 0 for embedding lookup
    ids_for_embed = input_ids.clone()
    ids_for_embed[ids_for_embed == IMAGE_TOKEN_INDEX] = 0
    
    # 6. Get token embeddings
    inputs_embeds = model.llm.model.embed_tokens(ids_for_embed)
    
    # 7. Run LLM forward to get hidden states
    with torch.no_grad():
        outputs = model.llm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    
    # 8. Extract hidden states at placeholder positions
    hidden_states = outputs[0]  # [B, T, D]
    
    # Find placeholder positions in each sequence
    ph_positions = []
    for i in range(2):  # cond and uncond
        # Find first placeholder position
        pos = (input_ids[i] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            start_pos = pos[0].item()
            ph_positions.append((start_pos, start_pos + n_positions))
        else:
            # Fallback if no placeholders found
            ph_positions.append((0, n_positions))
    
    # Extract embeddings at placeholder positions
    embed_from_body = []
    for i, (start, end) in enumerate(ph_positions):
        embed_from_body.append(hidden_states[i, start:end])
    
    embed_from_body = torch.stack(embed_from_body)  # [2, n_positions, D]
    
    # 9. Generate with RQ-Transformer
    with torch.no_grad():
        rq = model.vision_tower.vision_tower.rqtransformer
        rqvae = model.vision_tower.vision_tower.rqvaesiglip
        out_features, _ = rq.generate(embed_from_body, model_aux=rqvae, cfg=cfg)
    
    # 10. Decode with RQ-VAE
    H = W = int(n_positions ** 0.5)
    img_feats = out_features.reshape(2, H, W, -1)
    
    with torch.no_grad():
        images = rqvae.decode(img_feats).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255)
    
    # Return only conditional samples (first one)
    return images[0:batch_size]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A cat")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--generation_nums", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="generated_images/")
    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = vila_u.load(args.model_path)
    model = model.half()  # Use half precision to save memory
    model.eval()
    
    # Generate images
    print(f"Generating image with prompt: '{args.prompt}'")
    
    all_images = []
    for i in range(args.generation_nums):
        print(f"Generating image {i+1}/{args.generation_nums}")
        torch.cuda.empty_cache()  # Clear GPU cache between generations
        images = direct_generate_image(model, args.prompt, args.cfg, batch_size=1)
        all_images.append(images)
    
    if all_images:
        response = torch.cat(all_images, dim=0)
        save_image(response, args.save_path)
        print(f"Images saved to {args.save_path}")
    else:
        print("No images generated")