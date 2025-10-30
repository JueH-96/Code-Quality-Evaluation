import sys
import bisect

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    a_coords = list(map(int, sys.stdin.readline().split()))
    a_coords.sort()

    # Process each query
    for _ in range(Q):
        b_val, k_val = map(int, sys.stdin.readline().split())

        # Binary search for the k_val-th distance D
        # We are looking for the smallest D such that
        # count of A_i where |A_i - b_val| <= D is at least k_val.
        
        low_dist = 0
        # Max possible distance is abs(10^8 - (-10^8)) = 2 * 10^8.
        # Coordinates are integers, so distances are integers.
        high_dist = 2 * (10**8) 
        ans_dist = high_dist # Initialize with a value >= max possible distance

        while low_dist <= high_dist:
            mid_dist = low_dist + (high_dist - low_dist) // 2
            
            # Points A_i are sought in the coordinate range [b_val - mid_dist, b_val + mid_dist]
            min_coord_val_in_range = b_val - mid_dist
            max_coord_val_in_range = b_val + mid_dist
            
            # Count points A_i such that min_coord_val_in_range <= A_i <= max_coord_val_in_range
            # This is equivalent to |A_i - b_val| <= mid_dist
            
            # Find index of the first element >= min_coord_val_in_range
            left_idx = bisect.bisect_left(a_coords, min_coord_val_in_range)
            
            # Find index of the first element > max_coord_val_in_range
            # (This means elements up to right_idx-1 are <= max_coord_val_in_range)
            right_idx = bisect.bisect_right(a_coords, max_coord_val_in_range)
            
            count = right_idx - left_idx
            
            if count >= k_val:
                # mid_dist is a potential answer (or an even smaller D is possible)
                ans_dist = mid_dist
                high_dist = mid_dist - 1 # Try to find a smaller D
            else:
                # mid_dist is too small, D must be larger
                low_dist = mid_dist + 1
        
        print(ans_dist)

solve()