# File: inference/greedy_decode.py (V10.0: Gestalt-Aware with 6-Value Unpack Fix)

import torch
import numpy as np
import random
import os

# --- Import KG & Location ---
try:
    from models.kg import PoetryKnowledgeGraph
except ImportError:
    print("[Error] Could not import PoetryKnowledgeGraph. Make sure models/kg.py is accessible.")
    PoetryKnowledgeGraph = None

# Import Location Generator
try:
    from models.location import LocationSignalGenerator
except ImportError:
    print("[Error] Could not import LocationSignalGenerator. Make sure models/location.py is accessible.")
    LocationSignalGenerator = None

# [NEW] Import Integrated Visualization Tool
try:
    from data.visualize import draw_integrated_heatmap
except ImportError:
    draw_integrated_heatmap = None
# -----------------

# [V8.0] Relaxed Shape Priors (Data-Driven Approach)
CLASS_SHAPE_PRIORS = {
    2: {'min_w': 0.05, 'min_h': 0.05}, # Mountain (山)
    3: {'min_w': 0.05, 'min_h': 0.02}, # Water (水)
    4: {'min_w': 0.01, 'min_h': 0.02}, # People (人)
    5: {'min_w': 0.02, 'min_h': 0.05}, # Tree (树)
    6: {'min_w': 0.03, 'min_h': 0.03}, # Building (楼)
    7: {'min_w': 0.05, 'min_h': 0.02}, # Bridge (桥)
    8: {'min_w': 0.01, 'min_h': 0.01}, # Flower (花)
    9: {'min_w': 0.01, 'min_h': 0.01}, # Bird (鸟)
    10: {'min_w': 0.02, 'min_h': 0.02} # Animal (兽)
}

# Class ID Mapping
CLASS_ID_TO_NAME = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def greedy_decode_poem_layout(model, tokenizer, poem: str, max_elements=None, device='cuda', mode='greedy', top_k=3):
    """
    Query-Based decoding with Location Guidance & CVAE Diversity.
    [Updated V10.0] Fixed unpacking for 6-value return (added aux_outputs).
    
    Args:
        model: Trained Poem2LayoutGenerator (V10.0 with text-to-gestalt alignment)
        tokenizer: BertTokenizer
        poem: Input poem string
        max_elements: Max number of objects to generate
        device: 'cuda' or 'cpu'
        mode: 'greedy' or 'sample'
        top_k: Top-K sampling for location generation
        
    Returns:
        layout: List of tuples [(cls_id, cx, cy, w, h, bx, by, rot, flow), ...]
                Note: 9-element tuple includes class_id + 4D coords + 4D gestalt
    """
    if PoetryKnowledgeGraph is None:
        print("[Warning] PoetryKnowledgeGraph not available, returning empty layout.")
        return []

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    
    # =========================================================================
    # 1. Instantiate Components
    # =========================================================================
    pkg = PoetryKnowledgeGraph()
    
    if LocationSignalGenerator is not None:
        location_gen = LocationSignalGenerator(grid_size=8)
    else:
        location_gen = None
    
    # =========================================================================
    # 2. Extract KG Content
    # =========================================================================
    kg_vector = pkg.extract_visual_feature_vector(poem)
    kg_vector_t = torch.as_tensor(kg_vector)
    if kg_vector_t.device != device:
        kg_vector_t = kg_vector_t.cpu()
        
    existing_indices = torch.nonzero(kg_vector_t > 0).squeeze(1)
    raw_class_ids = (existing_indices + 2).tolist()
    
    if not raw_class_ids:
        print(f"[Warning] No objects extracted from poem: '{poem}'")
        return []
        
    # KG Quantity Expansion
    if hasattr(pkg, 'expand_ids_with_quantity'):
        kg_class_ids = pkg.expand_ids_with_quantity(raw_class_ids, poem)
    else:
        kg_class_ids = raw_class_ids
        
    if max_elements:
        kg_class_ids = kg_class_ids[:max_elements]
    
    print(f"[Inference] Extracted objects: {kg_class_ids}")
        
    # =========================================================================
    # 3. Prepare Model Inputs
    # =========================================================================
    kg_class_tensor = torch.tensor([kg_class_ids], dtype=torch.long).to(device)
    
    # Build Spatial Matrix
    try:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem, obj_ids=kg_class_ids)
    except TypeError:
        kg_spatial_matrix_np = pkg.extract_spatial_matrix(poem)
        
    kg_spatial_matrix = torch.as_tensor(kg_spatial_matrix_np, dtype=torch.long).unsqueeze(0).to(device)
    
    # =========================================================================
    # 4. Generate Location Guidance Signals
    # =========================================================================
    location_grids_tensor = None
    heatmap_layers = [] 
    
    if location_gen is not None:
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32)
        grids_list = []
        
        for i, cls_id in enumerate(kg_class_ids):
            mat_len = kg_spatial_matrix_np.shape[0]
            if i < mat_len:
                row = kg_spatial_matrix_np[i]
                col = kg_spatial_matrix_np[:, i]
            else:
                row = np.zeros(mat_len)
                col = np.zeros(mat_len)
            
            signal, current_occupancy = location_gen.infer_stateful_signal(
                i, row, col, current_occupancy, 
                mode=mode, top_k=top_k 
            )
            
            # [Optional] Random shift for sampling mode
            if mode == 'sample' and random.random() < 0.6:
                shift_val = random.randint(-2, 2)
                signal = torch.roll(signal, shifts=shift_val, dims=1)
            
            grids_list.append(signal)
            if draw_integrated_heatmap is not None:
                heatmap_layers.append((signal.cpu().numpy(), int(cls_id)))
            
        location_grids_tensor = torch.stack(grids_list).unsqueeze(0).to(device)

        # [Optional] Save integrated heatmap visualization
        if draw_integrated_heatmap is not None and len(heatmap_layers) > 0:
            safe_poem_name = "".join(x for x in poem if x.isalnum())[:10]
            save_dir = os.path.join("outputs", "heatmaps")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"integrated_{safe_poem_name}.png")
            draw_integrated_heatmap(heatmap_layers, poem, save_path)
            print(f"  -> Heatmap saved: {save_path}")
    
    # =========================================================================
    # 5. Forward Pass
    # =========================================================================
    inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=64)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    padding_mask = torch.zeros(kg_class_tensor.shape, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        # =====================================================================
        # [V10.0 CRITICAL FIX] Model now returns 6 values:
        # (mu, logvar, dynamic_layout, decoder_output, pred_heatmaps, aux_outputs)
        # 
        # Return values breakdown:
        # - mu, logvar: CVAE latent distribution parameters
        # - dynamic_layout: [B, N, 8] (4D coords + 4D gestalt)
        # - decoder_output: [B, N, hidden_size] decoder features
        # - pred_heatmaps: [B, N, 64, 64] spatial heatmaps
        # - aux_outputs: dict with gestalt_decoder, gestalt_text, gestalt_gate
        # =====================================================================
        _, _, pred_boxes, _, _, _ = model(  # ✅ Changed from 5 to 6 underscores
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            kg_class_ids=kg_class_tensor, 
            padding_mask=padding_mask, 
            kg_spatial_matrix=kg_spatial_matrix, 
            location_grids=location_grids_tensor,
            target_boxes=None  # Inference mode, no GT
        )
        
    # =========================================================================
    # 6. Format Output
    # =========================================================================
    layout = []
    boxes_flat = pred_boxes[0].cpu().tolist()  # [N, 8]
    
    for cls_id, box in zip(kg_class_ids, boxes_flat):
        cid = int(cls_id)
        
        # Extract coordinates (first 4 dimensions)
        if len(box) >= 4:
            cx, cy, w, h = box[:4] 
        else:
            cx, cy, w, h = 0.5, 0.5, 0.1, 0.1

        # Apply minimum size constraints
        w, h = max(w, 0.01), max(h, 0.01)
        
        if cid in CLASS_SHAPE_PRIORS:
            prior = CLASS_SHAPE_PRIORS[cid]
            if 'min_w' in prior: 
                w = max(w, prior['min_w'])
            if 'min_h' in prior: 
                h = max(h, prior['min_h'])
        
        # Extract gestalt parameters (last 4 dimensions)
        if len(box) >= 8:
            bias_x, bias_y, rotation, flow = box[4:8]
            gestalt_params = [bias_x, bias_y, rotation, flow]
        else:
            gestalt_params = [0.0, 0.0, 0.0, 0.0]

        # Build final tuple: (cls_id, cx, cy, w, h, bias_x, bias_y, rotation, flow)
        full_item = [float(cid), cx, cy, w, h] + gestalt_params
        layout.append(tuple(full_item))
    
    print(f"[Inference] Generated {len(layout)} objects")
    
    return layout


# =============================================================================
# [NEW V10.0] 辅助函数：解析和可视化态势信息
# =============================================================================
def analyze_gestalt_from_layout(layout):
    """
    分析布局中的态势信息
    
    Args:
        layout: List of tuples [(cls_id, cx, cy, w, h, bias_x, bias_y, rotation, flow), ...]
    
    Returns:
        dict: 态势统计信息
    """
    if not layout:
        return {}
    
    gestalt_stats = {
        'flow_mean': 0.0,
        'flow_std': 0.0,
        'rotation_mean': 0.0,
        'dry_objects': [],    # flow < -0.3
        'wet_objects': [],    # flow > 0.3
        'tilted_objects': []  # |rotation| > 0.2
    }
    
    flows = []
    rotations = []
    
    for item in layout:
        if len(item) >= 9:
            cls_id, cx, cy, w, h, bias_x, bias_y, rotation, flow = item
            flows.append(flow)
            rotations.append(rotation)
            
            class_name = CLASS_ID_TO_NAME.get(int(cls_id), "unknown")
            
            if flow < -0.3:
                gestalt_stats['dry_objects'].append((class_name, flow))
            if flow > 0.3:
                gestalt_stats['wet_objects'].append((class_name, flow))
            if abs(rotation) > 0.2:
                gestalt_stats['tilted_objects'].append((class_name, rotation))
    
    if flows:
        gestalt_stats['flow_mean'] = np.mean(flows)
        gestalt_stats['flow_std'] = np.std(flows)
    if rotations:
        gestalt_stats['rotation_mean'] = np.mean(rotations)
    
    return gestalt_stats


def print_gestalt_analysis(layout, poem):
    """
    打印态势分析结果
    
    Args:
        layout: 布局列表
        poem: 诗歌文本
    """
    stats = analyze_gestalt_from_layout(layout)
    
    print("\n" + "="*60)
    print(f">>> GESTALT ANALYSIS FOR: '{poem}' <<<")
    print("="*60)
    
    if not stats:
        print("No gestalt information available.")
        return
    
    print(f"Flow Statistics:")
    print(f"  Mean: {stats['flow_mean']:.3f}")
    print(f"  Std:  {stats['flow_std']:.3f}")
    print(f"  Rotation Mean: {stats['rotation_mean']:.3f}")
    
    if stats['dry_objects']:
        print(f"\n干涩物体 (Dry/Flying White):")
        for name, flow in stats['dry_objects']:
            print(f"  - {name}: flow={flow:.3f}")
    
    if stats['wet_objects']:
        print(f"\n湿润物体 (Wet/Diffusion):")
        for name, flow in stats['wet_objects']:
            print(f"  - {name}: flow={flow:.3f}")
    
    if stats['tilted_objects']:
        print(f"\n倾斜物体 (Tilted):")
        for name, rot in stats['tilted_objects']:
            print(f"  - {name}: rotation={rot:.3f}")
    
    print("="*60 + "\n")