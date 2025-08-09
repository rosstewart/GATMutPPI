#!/usr/bin/env python3
import sys
import numpy as np
import os

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print(f"Error: Expected 3 arguments, got {len(sys.argv) - 1}")
        print("Usage: python script.py arr1.npy arr2.npy mask.npy")
        sys.exit(1)
    
    # Load arrays
    arr1_path = sys.argv[1]
    arr2_path = sys.argv[2]
    mask_path = sys.argv[3]
    
    print(f"Loading arrays...")
    arr1 = np.load(arr1_path)
    arr2 = np.load(arr2_path)
    mask = np.load(mask_path)
    
    # Validate mask contains only 1s, 2s, and 3s
    unique_values = np.unique(mask)
    expected_values = {1, 2, 3}
    if not set(unique_values).issubset(expected_values):
        raise ValueError(f"Mask contains invalid values: {unique_values}. Expected only 1, 2, and 3.")
    
    # Validate all arrays have the same length
    if not (len(arr1) == len(arr2) == len(mask)):
        raise ValueError(f"Arrays have different lengths: arr1={len(arr1)}, arr2={len(arr2)}, mask={len(mask)}")
    
    print(f"Arrays validated. Length: {len(arr1)}")
    print(f"Mask unique values: {unique_values}")
    
    # Process each array
    for arr_path, arr in [(arr1_path, arr1), (arr2_path, arr2)]:
        # Split by mask values and save
        for class_num in [1, 2, 3]:
            if class_num in unique_values:
                # Create subset
                subset = arr[mask == class_num]
                
                # Generate output filename
                base_path = arr_path.replace('.npy', '')
                output_path = f"{base_path}_c{class_num}.npy"
                
                # Save
                np.save(output_path, subset)
                print(f"Saved {len(subset)} elements to {output_path}")
            else:
                print(f"Warning: No elements with mask value {class_num} for {arr_path}")

if __name__ == "__main__":
    main()
