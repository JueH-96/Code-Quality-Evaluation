import sys
import math

def dist(p1, p2):
    """Calculates the Euclidean distance between two points."""
    # Ensure inputs are floats for sqrt and division, although int difference works fine
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((float(x1) - float(x2))**2 + (float(y1) - float(y2))**2)

def solve():
    # Read input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    S = float(line1[1])
    T = float(line1[2])

    segments = []
    all_endpoints = []
    for _ in range(N):
        x1, y1, x2, y2 = map(int, sys.stdin.readline().split())
        segments.append(((x1, y1), (x2, y2))) # Store as tuples of ints
        all_endpoints.append((x1, y1)) # Store as tuples of ints
        all_endpoints.append((x2, y2)) # Store as tuples of ints

    start_point = (0, 0) # Store as tuple of ints

    # dp[mask][location_idx] = min time to print segments in mask, ending at location_idx
    # location_idx: 0 to 2*N-1 maps to all_endpoints[location_idx]
    # location_idx: 2*N maps to start_point (0,0)

    dp_size = 2 * N + 1
    dp = [[float('inf')] * dp_size for _ in range(1 << N)]

    # Initial state: No segments printed (mask 0), laser at (0,0) (location_idx 2*N)
    dp[0][2*N] = 0.0 # Initialize with 0.0 as float

    # Iterate through masks from 0 up to (1<<N)-1
    # The order of mask iteration doesn't strictly matter for correctness if we iterate enough times
    # (like Bellman-Ford), but processing masks in increasing order of size ensures
    # dp[mask] values are finalized before used to compute dp[new_mask].
    # Simple iteration 0 to (1<<N)-1 works for this bitmask DP.
    for mask in range(1 << N):
        for current_loc_idx in range(dp_size):
            if dp[mask][current_loc_idx] == float('inf'):
                continue # State is unreachable

            # Get current position coordinates
            if current_loc_idx == 2 * N:
                current_pos = start_point
            else:
                current_pos = all_endpoints[current_loc_idx]

            # Try printing each unprinted segment
            for i in range(N):
                # If segment i is not printed yet
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)

                    p_i1 = segments[i][0] # Endpoint 1 of segment i (tuple of ints)
                    p_i2 = segments[i][1] # Endpoint 2 of segment i (tuple of ints)
                    segment_len = dist(p_i1, p_i2) # dist handles int/float conversion

                    # Calculate printing cost (fixed for this segment regardless of start endpoint)
                    print_cost = segment_len / T

                    # Option 1: Start printing segment i from p_i1 (all_endpoints[2*i])
                    travel_cost_to_p_i1 = dist(current_pos, p_i1) / S
                    total_step_cost_opt1 = travel_cost_to_p_i1 + print_cost
                    # Ends at p_i2 (all_endpoints[2*i+1])
                    new_location_idx_opt1 = 2 * i + 1

                    # Update DP table entry for the new state
                    dp[new_mask][new_location_idx_opt1] = min(dp[new_mask][new_location_idx_opt1], dp[mask][current_loc_idx] + total_step_cost_opt1)

                    # Option 2: Start printing segment i from p_i2 (all_endpoints[2*i+1])
                    travel_cost_to_p_i2 = dist(current_pos, p_i2) / S
                    total_step_cost_opt2 = travel_cost_to_p_i2 + print_cost
                    new_location_idx_opt2 = 2 * i # Ends at p_i1 (all_endpoints[2*i])

                    # Update DP table entry
                    dp[new_mask][new_location_idx_opt2] = min(dp[new_mask][new_location_idx_opt2], dp[mask][current_loc_idx] + total_step_cost_opt2)

    # Find the minimum time among all states where all segments are printed
    final_mask = (1 << N) - 1
    min_total_time = float('inf')

    # The laser must end at one of the 2N endpoints (indices 0 to 2N-1 in DP state)
    for location_idx in range(2 * N):
         min_total_time = min(min_total_time, dp[final_mask][location_idx])

    # Output the result with required precision
    print(f"{min_total_time:.20f}")

# Execute the solver function
solve()