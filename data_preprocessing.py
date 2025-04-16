# # import argparse
# # from pycocotools.coco import COCO
# # import numpy as np
# # import os
# # import cv2
# # from pathlib import Path
# # from tqdm import tqdm

# # def main():
# #     parser = argparse.ArgumentParser(description='COCO Mask Preprocessing')
# #     parser.add_argument('--annotations', type=str, required=True,
# #                         help='Path to COCO annotations file')
# #     parser.add_argument('--images', type=str, required=True,
# #                         help='Path to images directory')
# #     parser.add_argument('--output', type=str, required=True,
# #                         help='Output directory for masks')
# #     parser.add_argument('--max-images', type=int, default=5000,
# #                         help='Maximum number of images to process')
# #     args = parser.parse_args()

# #     # Create output directories
# #     Path(args.output).mkdir(parents=True, exist_ok=True)
    
# #     # Initialize COCO API
# #     coco = COCO(args.annotations)
    
# #     # Get all category IDs
# #     cat_ids = coco.getCatIds()
    
# #     # Get image IDs (limited to max-images)
# #     img_ids = coco.getImgIds()[:args.max_images]
    
# #     # Process images
# #     for img_id in tqdm(img_ids, desc="Processing images"):
# #         img_info = coco.loadImgs(img_id)[0]
# #         img_path = Path(args.images) / img_info['file_name']
        
# #         # Skip if image file doesn't exist
# #         if not img_path.exists():
# #             print(f"\nWarning: Missing image {img_path} - skipping")
# #             continue
            
# #         # Load annotations
# #         ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
# #         anns = coco.loadAnns(ann_ids)
        
# #         # Create empty mask
# #         mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        
# #         # Combine annotations
# #         for ann in anns:
# #             if ann["iscrowd"]:
# #                 mask += coco.annToMask(ann)
# #             else:
# #                 mask = np.maximum(mask, coco.annToMask(ann)*ann['category_id'])
        
# #         # Save mask
# #         mask_path = Path(args.output) / f"{Path(img_info['file_name']).stem}.png"
# #         cv2.imwrite(str(mask_path), mask)

# # if __name__ == "__main__":
# #     main()
    
    
# import argparse
# from pycocotools.coco import COCO
# import numpy as np
# import os
# import cv2
# from pathlib import Path
# from tqdm import tqdm

# def main():
#     parser = argparse.ArgumentParser(description='COCO Mask Preprocessing')
#     parser.add_argument('--annotations', type=str, required=True,
#                         help='Path to COCO annotations file')
#     parser.add_argument('--images', type=str, required=True,
#                         help='Path to images directory')
#     parser.add_argument('--output', type=str, required=True,
#                         help='Output directory for masks')
#     parser.add_argument('--max-images', type=int, default=5000,
#                         help='Maximum number of images to process')
#     args = parser.parse_args()

#     # Create output directory
#     Path(args.output).mkdir(parents=True, exist_ok=True)
    
#     coco = COCO(args.annotations)
#     cat_ids = coco.getCatIds()
#     img_ids = coco.getImgIds()[:args.max_images]
    
#     for img_id in tqdm(img_ids, desc="Processing images"):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = Path(args.images) / img_info['file_name']
        
#         if not img_path.exists():
#             print(f"\n[Warning] Missing image: {img_path}")
#             continue
        
#         ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
#         anns = coco.loadAnns(ann_ids)

#         # Skip if there are no valid annotations
#         if not anns:
#             print(f"\n[Warning] No annotations found for image: {img_info['file_name']}")
#             continue

#         mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

#         # for ann in anns:
#         #     try:
#         #         ann_mask = coco.annToMask(ann)
#         #         if ann["iscrowd"]:
#         #             # Overlay crowd masks (use class 255)
#         #             mask = np.maximum(mask, ann_mask.astype(np.uint8) * 255)
#         #         else:
#         #             category_id = ann['category_id']
#         #             # Retain max category ID in case of overlap
#         #             mask = np.where(ann_mask, np.maximum(mask, category_id), mask)
#         #     except Exception as e:
#         #         print(f"\n[Error] Failed processing annotation ID {ann['id']}: {e}")
#         #         continue
#         for ann in anns:
#             if ann["iscrowd"]:
#                 continue  # Skip crowd masks entirely
#             try:
#                 ann_mask = coco.annToMask(ann)
#                 category_id = ann['category_id']
#                 mask = np.where(ann_mask, np.maximum(mask, category_id), mask)
#             except Exception as e:
#                 print(f"\n[Error] Failed processing annotation ID {ann['id']}: {e}")
#                 continue

#         mask_path = Path(args.output) / f"{Path(img_info['file_name']).stem}.png"
#         cv2.imwrite(str(mask_path), mask)

# if __name__ == "__main__":
#     main()
    
import argparse
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='COCO Mask Preprocessing')
    parser.add_argument('--annotations', type=str, required=True,
                        help='Path to COCO annotations file')
    parser.add_argument('--images', type=str, required=True,
                        help='Path to images directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--max-images', type=int, default=5000,
                        help='Maximum number of images to process')
    args = parser.parse_args()

    # Create organized directory structure
    images_dir = Path(args.output) / 'images'
    masks_dir = Path(args.output) / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    coco = COCO(args.annotations)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds()[:args.max_images]
    
    processed_count = 0
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(args.images) / img_info['file_name']
        
        # Skip if image file doesn't exist
        if not img_path.exists():
            print(f"\n[Warning] Missing image: {img_path}")
            continue
        
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)

        # Skip if there are no annotations
        if not anns:
            print(f"\n[Warning] No annotations found for image: {img_info['file_name']}")
            continue
            
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        has_valid_annotation = False

        for ann in anns:
            if ann["iscrowd"]:
                continue  # Skip crowd annotations
            try:
                ann_mask = coco.annToMask(ann)
                category_id = ann['category_id']
                mask = np.where(ann_mask, np.maximum(mask, category_id), mask)
                has_valid_annotation = True
            except Exception as e:
                print(f"\n[Error] Failed processing annotation ID {ann['id']}: {e}")
                continue
        
        # Skip if no valid annotations after processing
        if not has_valid_annotation:
            print(f"\n[Warning] No valid annotations for image: {img_info['file_name']}")
            continue
        
        # Create base filename
        base_filename = Path(img_info['file_name']).stem
        
        # Copy original image
        dest_img_path = images_dir / f"{base_filename}.jpg"
        shutil.copy(img_path, dest_img_path)
        
        # Save mask
        mask_path = masks_dir / f"{base_filename}.png"
        cv2.imwrite(str(mask_path), mask)
        
        processed_count += 1

    print(f"\nSuccessfully processed {processed_count} image-mask pairs")

if __name__ == "__main__":
    main()