import os
import shutil
from pathlib import Path


def organize_voc2007(voc_dir):
    """
    Organize VOC2007 images into train and test folders based on split files
    
    Args:
        voc_dir: Path to VOC2007 directory containing JPEGImages, Annotations, ImageSets
    """
    voc_path = Path(voc_dir)
    
    # Check if required directories exist
    images_dir = voc_path / "JPEGImages"
    annotations_dir = voc_path / "Annotations"
    imagesets_dir = voc_path / "ImageSets" / "Main"
    
    if not images_dir.exists():
        print(f"Error: {images_dir} not found!")
        return
    
    if not imagesets_dir.exists():
        print(f"Error: {imagesets_dir} not found!")
        print("ImageSets/Main directory is required for train/test splits")
        return
    
    # Create train and test directories
    train_dir = voc_path / "train"
    test_dir = voc_path / "test"
    
    train_images_dir = train_dir / "images"
    train_annotations_dir = train_dir / "annotations"
    test_images_dir = test_dir / "images"
    test_annotations_dir = test_dir / "annotations"
    
    # Create all directories
    for d in [train_images_dir, train_annotations_dir, test_images_dir, test_annotations_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Organizing VOC2007 into train/test folders")
    print("="*60)
    
    # Function to copy files based on split
    def copy_split(split_file, dest_images, dest_annotations):
        if not split_file.exists():
            print(f"Warning: {split_file} not found, skipping...")
            return 0
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f if line.strip()]
        
        copied = 0
        for img_id in image_ids:
            # Copy image
            src_img = images_dir / f"{img_id}.jpg"
            dst_img = dest_images / f"{img_id}.jpg"
            
            if src_img.exists():
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
                copied += 1
            else:
                print(f"Warning: Image {src_img} not found")
            
            # Copy annotation if it exists
            if annotations_dir.exists():
                src_ann = annotations_dir / f"{img_id}.xml"
                dst_ann = dest_annotations / f"{img_id}.xml"
                
                if src_ann.exists() and not dst_ann.exists():
                    shutil.copy2(src_ann, dst_ann)
        
        return copied
    
    # Process train split (train + val combined for training)
    print("\nProcessing training set...")
    train_file = imagesets_dir / "train.txt"
    val_file = imagesets_dir / "val.txt"
    
    train_count = copy_split(train_file, train_images_dir, train_annotations_dir)
    val_count = copy_split(val_file, train_images_dir, train_annotations_dir)
    
    total_train = train_count + val_count
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Total training: {total_train} images")
    
    # Process test split
    print("\nProcessing test set...")
    test_file = imagesets_dir / "test.txt"
    test_count = copy_split(test_file, test_images_dir, test_annotations_dir)
    print(f"  Test: {test_count} images")
    
    # Summary
    print("\n" + "="*60)
    print("✓ Organization Complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  {voc_path}/")
    print(f"    ├── train/")
    print(f"    │   ├── images/          ({total_train} images)")
    print(f"    │   └── annotations/     ({total_train} annotations)")
    print(f"    └── test/")
    print(f"        ├── images/          ({test_count} images)")
    print(f"        └── annotations/     ({test_count} annotations)")
    print(f"\nOriginal JPEGImages folder preserved")
    print(f"Total images organized: {total_train + test_count}")


def organize_simple(voc_dir, train_ratio=0.8):
    """
    Simple organization without ImageSets (creates random split)
    Use this only if ImageSets directory doesn't exist
    
    Args:
        voc_dir: Path to VOC2007 directory
        train_ratio: Ratio of images for training (default 0.8 = 80% train, 20% test)
    """
    import random
    
    voc_path = Path(voc_dir)
    images_dir = voc_path / "JPEGImages"
    
    if not images_dir.exists():
        print(f"Error: {images_dir} not found!")
        return
    
    # Get all image files
    all_images = list(images_dir.glob("*.jpg"))
    random.shuffle(all_images)
    
    # Split into train/test
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]
    
    print("="*60)
    print(f"Creating random split ({int(train_ratio*100)}% train, {int((1-train_ratio)*100)}% test)")
    print("="*60)
    
    # Create directories
    train_dir = voc_path / "train" / "images"
    test_dir = voc_path / "test" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train images
    print(f"\nCopying {len(train_images)} training images...")
    for img in train_images:
        shutil.copy2(img, train_dir / img.name)
    
    # Copy test images
    print(f"Copying {len(test_images)} test images...")
    for img in test_images:
        shutil.copy2(img, test_dir / img.name)
    
    # Copy annotations if they exist
    annotations_dir = voc_path / "Annotations"
    if annotations_dir.exists():
        train_ann_dir = voc_path / "train" / "annotations"
        test_ann_dir = voc_path / "test" / "annotations"
        train_ann_dir.mkdir(parents=True, exist_ok=True)
        test_ann_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nCopying annotations...")
        for img in train_images:
            ann_file = annotations_dir / f"{img.stem}.xml"
            if ann_file.exists():
                shutil.copy2(ann_file, train_ann_dir / ann_file.name)
        
        for img in test_images:
            ann_file = annotations_dir / f"{img.stem}.xml"
            if ann_file.exists():
                shutil.copy2(ann_file, test_ann_dir / ann_file.name)
    
    print("\n✓ Random split complete!")
    print(f"  Train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")


if __name__ == "__main__":
    import sys
    
    # Default path
    voc_dir = "data/VOC2007"
    
    # Check if path provided as argument
    if len(sys.argv) > 1:
        voc_dir = sys.argv[1]
    
    voc_path = Path(voc_dir)
    
    if not voc_path.exists():
        print(f"Error: Directory {voc_dir} not found!")
        print(f"\nUsage: python organize_voc2007.py [path_to_voc2007]")
        print(f"Example: python organize_voc2007.py data/VOC2007")
        sys.exit(1)
    
    # Check if ImageSets exists for official split
    imagesets_dir = voc_path / "ImageSets" / "Main"
    
    if imagesets_dir.exists():
        print("Found ImageSets directory - using official train/test split")
        organize_voc2007(voc_dir)
    else:
        print("ImageSets directory not found - creating random 80/20 split")
        user_input = input("Continue with random split? (y/n): ")
        if user_input.lower() == 'y':
            organize_simple(voc_dir, train_ratio=0.8)
        else:
            print("Cancelled. Please ensure VOC2007 has ImageSets/Main directory")