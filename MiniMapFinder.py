import os
import pandas as pd
import argparse
import gc
import numpy as np


def create_minimaps(Z, N):
    # Define input and output file paths
    input_file = f"WholeMaps/{Z}_{N}_6D_Whole_Map.txt"
    output_file_6d = f"MiniMaps/{Z}_{N}_6D_B20B30_MiniMap.txt"
    output_file_4d = f"MiniMaps/{Z}_{N}_4D_B20B30_MiniMap.txt"
    output_file_6d_starting = f"MiniMaps/{Z}_{N}_6D_Starting_MiniMap.txt"
    output_file_4d_starting = f"MiniMaps/{Z}_{N}_4D_Starting_MiniMap.txt"
    output_file_6d_fusion = f"MiniMaps/{Z}_{N}_6D_Fusion_MiniMap.txt"
    output_file_4d_fusion = f"MiniMaps/{Z}_{N}_4D_Fusion_MiniMap.txt"
    # New output files for B10-constant minimaps
    output_file_6d_b10_const = f"MiniMaps/{Z}_{N}_6D_B10const_MiniMap.txt"
    output_file_4d_b10_const = f"MiniMaps/{Z}_{N}_4D_B10const_MiniMap.txt"

    # Create the MiniMaps directory if it doesn't exist
    os.makedirs("MiniMaps", exist_ok=True)

    # Read the WholeMap file
    print(f"Reading WholeMap file: {input_file}")
    df = pd.read_csv(input_file, sep=' ', header=None,
                     names=['M', 'E', 'ELD', 'ESH', 'B10', 'B20', 'B30', 'B40', 'B50', 'B60'])

    # Filter the data for the specified ranges of B20 and B30
    filtered_points_6d = df[
        (df['B20'] >= 0.0) & (df['B20'] <= 2.0) &
        (df['B30'] >= -0.5) & (df['B30'] <= 0.5)
        ]

    # Additional filter for 4D MiniMap
    filtered_points_4d = filtered_points_6d[
        (filtered_points_6d['B50'] == 0.0) &
        (filtered_points_6d['B60'] == 0.0)
        ]

    # Free memory
    del df
    gc.collect()

    # Define starting points
    starting_points = {
        (102, 154): [0.942, 1.754, 0.002, -0.291, -0.001, 0.076],
        (104, 154): [0.917, 1.754, 0.002, -0.291, -0.001, 0.076],
        (106, 156): [0.869, 1.754, 0.002, -0.291, -0.001, 0.076],
        (108, 158): [0.824, 1.754, 0.002, -0.291, -0.001, 0.076],
        (110, 162): [0.763, 1.754, 0.002, -0.291, -0.001, 0.076],
        (112, 166): [0.706, 1.755, 0.002, -0.292, -0.001, 0.077]
    }

    def process_and_save_minimap(points, output_file, groupby_cols=['B20', 'B30']):
        # Group by specified columns, find the index of minimum E for each group
        idx = points.groupby(groupby_cols)['E'].idxmin()

        # Use the index to select the rows with minimum E for each group
        minimap = points.loc[idx]

        # Sort the minimap by the groupby columns
        minimap = minimap.sort_values(groupby_cols)

        # Write minimap to output file using chunks
        chunk_size = 10_000  # Adjust based on available memory

        print(f"Writing MiniMap to: {output_file}")
        with open(output_file, 'w', newline='') as f:
            for chunk_start in range(0, len(minimap), chunk_size):
                df_chunk = minimap.iloc[chunk_start:chunk_start + chunk_size]
                df_chunk.to_csv(f, sep=' ', columns=['M', 'E', 'ELD', 'ESH', 'B10', 'B20', 'B30', 'B40', 'B50', 'B60'],
                                float_format='%.3f', header=False, index=False, mode='a', lineterminator='\n')

        print(f"MiniMap saved to {output_file}")
        print(f"Total points in MiniMap: {len(minimap)}")

    def create_starting_minimap(points, output_file, starting_point, is_4d=False):
        # Round the starting point values to the nearest 0.05
        rounded_starting_point = [round(x * 20) / 20 for x in starting_point]

        # Create a boolean mask for each column
        if is_4d:
            masks = [
                (points['B10'] == rounded_starting_point[0]) &
                (points['B40'] == rounded_starting_point[3])
            ]
        else:
            masks = [
                (points['B10'] == rounded_starting_point[0]) &
                (points['B40'] == rounded_starting_point[3]) &
                (points['B50'] == rounded_starting_point[4]) &
                (points['B60'] == rounded_starting_point[5])
            ]

        # Combine all masks
        final_mask = np.logical_and.reduce(masks)

        # Apply the mask to get the starting minimap
        starting_minimap = points[final_mask]

        # Sort the starting minimap
        starting_minimap = starting_minimap.sort_values(['B20', 'B30'])

        print(f"Writing Starting MiniMap to: {output_file}")
        starting_minimap.to_csv(output_file, sep=' ', columns=['M', 'E', 'ELD', 'ESH', 'B10', 'B20', 'B30', 'B40', 'B50', 'B60'],
                                float_format='%.3f', header=False, index=False)

        print(f"Starting MiniMap saved to {output_file}")
        print(f"Total points in Starting MiniMap: {len(starting_minimap)}")

    def create_fusion_minimap(points, output_file, starting_point):
        # Round the B10 starting point value to the nearest 0.05
        rounded_b10 = round(starting_point[0] * 20) / 20

        # Create a boolean mask for B10
        mask = (points['B10'] == rounded_b10) & (points['B40'] == 0.0) & (points['B50'] == 0.0) & (points['B60'] == 0.0)

        # Apply the mask to get the fusion minimap
        fusion_minimap = points[mask]

        # Sort the fusion minimap
        fusion_minimap = fusion_minimap.sort_values(['B20', 'B30'])

        print(f"Writing Fusion MiniMap to: {output_file}")
        fusion_minimap.to_csv(output_file, sep=' ', columns=['M', 'E', 'ELD', 'ESH', 'B10', 'B20', 'B30', 'B40', 'B50', 'B60'],
                              float_format='%.3f', header=False, index=False)

        print(f"Fusion MiniMap saved to {output_file}")
        print(f"Total points in Fusion MiniMap: {len(fusion_minimap)}")

    def create_b10_constant_minimaps(points_6d, points_4d, output_file_6d, output_file_4d):
        # Get unique B10 values
        b10_values = points_6d['B10'].unique()
        print(f"Processing {len(b10_values)} unique B10 values")

        # Process each B10 value
        for b10_val in b10_values:
            # Filter points for current B10 value
            points_6d_b10 = points_6d[points_6d['B10'] == b10_val]
            points_4d_b10 = points_4d[points_4d['B10'] == b10_val]

            # Create output filenames with B10 value
            output_6d = output_file_6d.replace('.txt', f'_B10_{b10_val:.3f}.txt')
            output_4d = output_file_4d.replace('.txt', f'_B10_{b10_val:.3f}.txt')

            # Process and save minimaps for this B10 value
            if len(points_6d_b10) > 0:
                process_and_save_minimap(points_6d_b10, output_6d)
            if len(points_4d_b10) > 0:
                process_and_save_minimap(points_4d_b10, output_4d)

    # Process and save regular minimaps
    process_and_save_minimap(filtered_points_6d, output_file_6d)
    process_and_save_minimap(filtered_points_4d, output_file_4d)

    # Process and save B10-constant minimaps
    create_b10_constant_minimaps(filtered_points_6d, filtered_points_4d,
                                 output_file_6d_b10_const, output_file_4d_b10_const)

    # Create and save starting minimaps if starting points exist
    if (Z, N) in starting_points:
        create_starting_minimap(filtered_points_6d, output_file_6d_starting, starting_points[(Z, N)])
        create_starting_minimap(filtered_points_4d, output_file_4d_starting, starting_points[(Z, N)], is_4d=True)
        create_fusion_minimap(filtered_points_6d, output_file_6d_fusion, starting_points[(Z, N)])
        create_fusion_minimap(filtered_points_4d, output_file_4d_fusion, starting_points[(Z, N)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 6D and 4D MiniMaps from a WholeMap file, including B10-constant maps.")
    parser.add_argument("Protons", type=int, help="Proton number")
    parser.add_argument("Neutrons", type=int, help="Neutron number")
    args = parser.parse_args()

    create_minimaps(args.Protons, args.Neutrons)
