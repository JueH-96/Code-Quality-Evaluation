import math
import sys

# Function to solve the problem
def solve():
    # Read N, S, T from the first line of input
    N, S, T = map(int, sys.stdin.readline().split())

    # Read the N line segments
    segments_raw = []
    for _ in range(N):
        x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
        segments_raw.append(((x1, y1), (x2, y2)))

    # --- Preprocessing Step 1: Map unique coordinate points to integer indices ---
    # This helps in reducing the state space for the DP table.
    # points_list: Stores the actual (x, y) coordinates, indexed by their unique ID.
    # point_to_idx: Maps (x, y) coordinates to their unique integer ID.
    
    points_list = [(0, 0)]  # Start with the initial laser position (0,0) at index 0
    point_to_idx = {(0, 0): 0}
    
    # segment_endpoints_mapped_idx: Stores the endpoints of each segment as indices
    #                               into the 'points_list'.
    # Example: segment_endpoints_mapped_idx[i] = (idx_of_endpoint1, idx_of_endpoint2)
    #          for the i-th segment.
    segment_endpoints_mapped_idx = [] 

    for i in range(N):
        pA_raw, pC_raw = segments_raw[i] # Current segment's raw (x,y) endpoints

        # Add pA_raw to points_list if it's new, and get its unique index
        if pA_raw not in point_to_idx:
            point_to_idx[pA_raw] = len(points_list)
            points_list.append(pA_raw)
        idxA = point_to_idx[pA_raw]

        # Add pC_raw to points_list if it's new, and get its unique index
        if pC_raw not in point_to_idx:
            point_to_idx[pC_raw] = len(points_list)
            points_list.append(pC_raw)
        idxC = point_to_idx[pC_raw]
        
        # Store the mapped indices for this segment
        segment_endpoints_mapped_idx.append((idxA, idxC))

    NUM_UNIQUE_POINTS = len(points_list) # Total number of unique coordinate points

    # --- Preprocessing Step 2: Precompute the fixed time required to print each segment ---
    # This is dist(endpoint1, endpoint2) / T for each segment.
    segment_print_times = [0.0] * N
    for i in range(N):
        pA_raw, pC_raw = segments_raw[i]
        # Calculate Euclidean distance between the segment's two endpoints
        dist_segment = math.sqrt((pA_raw[0] - pC_raw[0])**2 + (pA_raw[1] - pC_raw[1])**2)
        segment_print_times[i] = dist_segment / T

    # --- Dynamic Programming Approach (Bitmask DP) ---
    # dp[mask][current_point_idx] represents the minimum time to print all segments
    # indicated by the 'mask' (where i-th bit is 1 if segment i is printed),
    # with the laser currently positioned at 'points_list[current_point_idx]'.
    
    # Initialize DP table with a very large value (infinity)
    dp = [[float('inf')] * NUM_UNIQUE_POINTS for _ in range(1 << N)]

    # Base case: At the start, no segments are printed (mask 0), and laser is at (0,0) (index 0).
    # The time taken is 0.
    dp[0][0] = 0.0

    # Iterate through all possible masks (from 0 up to 2^N - 1).
    # Masks are processed in increasing order, ensuring that dp values for smaller masks
    # (fewer segments printed) are computed before larger masks.
    for mask in range(1 << N):
        # Iterate through all possible current laser positions (unique points)
        for current_point_idx in range(NUM_UNIQUE_POINTS):
            # If the current state (mask, current_point_idx) is unreachable, skip it
            if dp[mask][current_point_idx] == float('inf'):
                continue

            current_pos_coords = points_list[current_point_idx] # Get actual (x,y) of current position

            # Try to print each unprinted segment
            for next_segment_idx in range(N):
                # Check if this 'next_segment_idx' is already printed in the current 'mask'
                if (mask >> next_segment_idx) & 1:
                    continue # Segment already printed, skip to next

                # Calculate the 'new_mask' after printing 'next_segment_idx'
                new_mask = mask | (1 << next_segment_idx)

                # Get the unique indices of the endpoints for 'next_segment_idx'
                pA_idx, pC_idx = segment_endpoints_mapped_idx[next_segment_idx]
                pA_coords = points_list[pA_idx] # Actual (x,y) of endpoint A
                pC_coords = points_list[pC_idx] # Actual (x,y) of endpoint C
                
                # Retrieve the precomputed time to print this segment (fixed cost)
                time_to_print_segment = segment_print_times[next_segment_idx]

                # Option 1: Move laser to pA_coords, print segment to pC_coords
                # Time to move from current_pos_coords to pA_coords (without emitting)
                time_to_move_to_A = math.sqrt((current_pos_coords[0] - pA_coords[0])**2 + \
                                              (current_pos_coords[1] - pA_coords[1])**2) / S
                
                # Total time for this sequence of operations
                total_time_option1 = dp[mask][current_point_idx] + time_to_move_to_A + time_to_print_segment
                
                # Update DP table: new_mask, laser ends at pC_idx
                dp[new_mask][pC_idx] = min(dp[new_mask][pC_idx], total_time_option1)

                # Option 2: Move laser to pC_coords, print segment to pA_coords
                # Time to move from current_pos_coords to pC_coords (without emitting)
                time_to_move_to_C = math.sqrt((current_pos_coords[0] - pC_coords[0])**2 + \
                                              (current_pos_coords[1] - pC_coords[1])**2) / S
                
                # Total time for this sequence of operations
                total_time_option2 = dp[mask][current_point_idx] + time_to_move_to_C + time_to_print_segment
                
                # Update DP table: new_mask, laser ends at pA_idx
                dp[new_mask][pA_idx] = min(dp[new_mask][pA_idx], total_time_option2)
                
    # --- Find the minimum time among all final states ---
    # After processing all masks, the minimum total time will be found in the states
    # where all segments have been printed (i.e., mask is (1 << N) - 1).
    # The laser can end at any of the unique points after the last segment is printed.
    
    min_total_time = float('inf')
    all_segments_printed_mask = (1 << N) - 1 # This mask has all N bits set to 1

    # Iterate through all possible ending points for the laser
    for k in range(NUM_UNIQUE_POINTS):
        min_total_time = min(min_total_time, dp[all_segments_printed_mask][k])

    # Print the result formatted to high precision as required
    sys.stdout.write(f"{min_total_time:.20f}
")

# Call the solve function to run the program
solve()