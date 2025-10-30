# YOUR CODE HERE
import math
import sys

def solve():
    # Read input: N = number of segments, S = speed without laser, T = speed with laser
    # N is the number of line segments
    # S is the speed when not emitting laser (travel speed)
    # T is the speed when emitting laser (printing speed)
    N, S, T = map(int, sys.stdin.readline().split())
    
    # Read segment endpoint coordinates
    # Store coordinates for each segment
    segments_coords = []
    for _ in range(N):
        segments_coords.append(list(map(int, sys.stdin.readline().split()))) 

    # Store endpoints and calculate segment lengths
    # We use a list `endpoints` where `endpoints[2*i]` is the first endpoint (A_i, B_i) 
    # and `endpoints[2*i + 1]` is the second endpoint (C_i, D_i) of segment i.
    # This gives each endpoint a unique index from 0 to 2N-1.
    endpoints = []
    lengths = []
    total_segment_length = 0.0

    for i in range(N):
        # Extract coordinates for segment i
        A_i, B_i, C_i, D_i = segments_coords[i]
        p1 = (A_i, B_i) # First endpoint of segment i
        p2 = (C_i, D_i) # Second endpoint of segment i
        
        # Add endpoints to the list
        endpoints.append(p1) # Corresponds to index 2*i
        endpoints.append(p2) # Corresponds to index 2*i + 1
        
        # Calculate the length of segment i using Euclidean distance
        # math.dist(p1, p2) calculates sqrt((x1-x2)^2 + (y1-y2)^2)
        length = math.dist(p1, p2)
        lengths.append(length)
        # Accumulate total length of all segments for printing time calculation
        total_segment_length += length

    # Initialize the DP table
    # dp[mask][k] stores the minimum total travel distance (time spent moving without laser)
    # required to print the set of segments represented by the bitmask `mask`.
    # The state signifies that the last segment printed was segment `k // 2`, 
    # and the laser finished at endpoint `k`.
    # The DP table size is (2^N) rows (for masks) and (2N) columns (for endpoints).
    # Initialize all entries to infinity, indicating states are initially unreachable.
    dp = [[float('inf')] * (2 * N) for _ in range(1 << N)]

    # Base cases: Handle printing the very first segment.
    # The laser starts at the origin (0,0).
    start_pos = (0, 0)
    
    # Handle N=0 case explicitly. The problem constraint says N >= 1,
    # but handling N=0 costs nothing and makes the code more robust.
    if N == 0:
        print(f"{0.0:.17f}") # No segments, no time needed.
        return

    # Initialize DP states for printing the first segment (mask has exactly one bit set)
    for i in range(N):
        p1_idx = 2*i       # Index for endpoint P_i = (A_i, B_i)
        p2_idx = 2*i + 1   # Index for endpoint Q_i = (C_i, D_i)
        p1 = endpoints[p1_idx] # Coordinate tuple for P_i
        p2 = endpoints[p2_idx] # Coordinate tuple for Q_i
        
        # The mask representing that only segment i has been printed
        mask = 1 << i
        
        # Option 1: Print segment i starting from P1, ending at P2 (endpoint index p2_idx).
        # To start printing from P1, the laser must first move from start_pos=(0,0) to P1.
        # This travel is without laser. Calculate the distance.
        dist_to_p1 = math.dist(start_pos, p1)
        # The state reached is (mask, p2_idx) with accumulated travel distance dist_to_p1.
        # Update dp[mask][p2_idx] with this distance. Use min in case this state could somehow
        # be reached differently (not possible for base cases, but good practice).
        dp[mask][p2_idx] = min(dp[mask][p2_idx], dist_to_p1) 
        
        # Option 2: Print segment i starting from P2, ending at P1 (endpoint index p1_idx).
        # To start printing from P2, the laser must first move from start_pos=(0,0) to P2.
        # Calculate this travel distance.
        dist_to_p2 = math.dist(start_pos, p2)
        # The state reached is (mask, p1_idx) with accumulated travel distance dist_to_p2.
        dp[mask][p1_idx] = min(dp[mask][p1_idx], dist_to_p2) 

    # Fill the DP table using dynamic programming
    # Iterate through all possible masks from 1 up to (2^N - 1).
    # A mask represents a subset of segments that have been printed.
    for mask in range(1, 1 << N):
        # For each mask, iterate through all possible endpoints k (0 to 2N-1).
        # This endpoint k represents the position where the laser finished printing the last segment.
        for k in range(2 * N):
            # Check if the state (mask, k) is reachable (i.e., has a finite minimum distance calculated so far)
            if dp[mask][k] == float('inf'):
                continue # Skip unreachable states

            # The current position of the laser is at endpoint k
            current_pos = endpoints[k]
            
            # Consider printing the next segment j, which is not yet included in the current mask
            for j in range(N):
                # Check if the j-th bit is NOT set in the mask (i.e., segment j hasn't been printed yet)
                if not ((mask >> j) & 1):
                    # If segment j hasn't been printed, we can print it next.
                    # Calculate the mask for the state after printing segment j: it includes all segments from 'mask' plus segment 'j'.
                    next_mask = mask | (1 << j)
                    
                    # Get indices and coordinates of the two endpoints for segment j
                    p1_j_idx = 2*j       # Index for endpoint P_j = (A_j, B_j)
                    p2_j_idx = 2*j + 1   # Index for endpoint Q_j = (C_j, D_j)
                    p1_j = endpoints[p1_j_idx] # Coordinates of P_j
                    p2_j = endpoints[p2_j_idx] # Coordinates of Q_j
                    
                    # Option 1: Print segment j starting from Pj, ending at Qj (final endpoint index p2_j_idx).
                    # To start printing at Pj, the laser must move from its current_pos (endpoint k) to Pj.
                    # This movement is without the laser on. Calculate the travel distance.
                    travel_dist1 = dp[mask][k] + math.dist(current_pos, p1_j)
                    # Update the minimum distance found so far for the resulting state (next_mask, p2_j_idx).
                    dp[next_mask][p2_j_idx] = min(dp[next_mask][p2_j_idx], travel_dist1)
                    
                    # Option 2: Print segment j starting from Qj, ending at Pj (final endpoint index p1_j_idx).
                    # To start printing at Qj, the laser moves from current_pos to Qj without laser.
                    # Calculate the travel distance.
                    travel_dist2 = dp[mask][k] + math.dist(current_pos, p2_j)
                    # Update the minimum distance found so far for the resulting state (next_mask, p1_j_idx).
                    dp[next_mask][p1_j_idx] = min(dp[next_mask][p1_j_idx], travel_dist2)

    # After filling the DP table, find the minimum total travel distance required to print all N segments.
    # This corresponds to the minimum value in the DP table for the final mask, which has all N bits set.
    min_total_travel_dist = float('inf')
    # The final mask representing all N segments printed is (1 << N) - 1.
    final_mask = (1 << N) - 1
    
    # Iterate through all possible final endpoints k after printing all segments.
    # The minimum value among dp[final_mask][k] for all k gives the overall minimum travel distance.
    for k in range(2 * N):
       min_total_travel_dist = min(min_total_travel_dist, dp[final_mask][k])

    # Final calculation: Total time required is the sum of:
    # 1. Total time spent printing all segments (total length / printing speed T)
    # 2. Total time spent moving between segments without laser (minimum total travel distance / travel speed S)
    
    # Check if a valid path was found. If N >= 1, min_total_travel_dist should be finite.
    if min_total_travel_dist == float('inf'):
         # This case should technically not be reached under problem constraints (N>=1, finite coordinates).
         # If it were reached, it might indicate disconnected parts or an issue.
         # However, the problem implies connectivity from origin is possible.
         # If this occurs, printing 0.0 might be a placeholder, but suggests an error.
         # Based on problem setup, we expect a finite minimum travel distance.
         pass # Assuming finite result is always found for N >= 1.

    # Calculate total time
    total_time = total_segment_length / T + min_total_travel_dist / S
    
    # Print the final answer formatted to the required precision (17 decimal places for safety)
    print(f"{total_time:.17f}")

# Execute the solve function to run the program
solve()