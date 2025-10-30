import sys

def solve():
    N = int(sys.stdin.readline())
    buildings = []
    for _ in range(N):
        X, H = map(int, sys.stdin.readline().split())
        buildings.append((X, H))

    def check(h_val):
        # Returns True if ALL buildings are visible from (0, h_val), False otherwise.
        
        # max_slope_so_far tracks the maximum slope (rise over run) from (0, h_val)
        # to the top of any building processed so far. This forms the upper envelope
        # of lines of sight that subsequent buildings must clear.
        # Initialize with -float('inf') so the first building (which has no preceding obstacles)
        # effectively needs to clear a required height of 0.
        max_slope_so_far = -float('inf') 

        for i in range(N):
            X_k, H_k = buildings[i]

            # Calculate the height at X_k that the line of sight (defined by max_slope_so_far)
            # from (0, h_val) would reach. This is the minimum y-coordinate that a point
            # on building k must have to be visible.
            needed_y_at_X_k = h_val + max_slope_so_far * X_k

            # If the calculated needed height is negative, it means the line of sight from 
            # (0, h_val) to the previous maximum obstacle is already below the ground (y=0) at X_k.
            # In this scenario, building k only needs to clear the ground, so the effective
            # needed height is 0.
            if needed_y_at_X_k < 0:
                needed_y_at_X_k = 0

            # If the actual height of building k is less than or equal to the required height
            # to clear previous obstacles, then building k is not visible.
            # (The problem states "does not intersect", so H_k <= needed_y_at_X_k implies intersection).
            if H_k <= needed_y_at_X_k:
                return False
            
            # If building k is visible, update max_slope_so_far.
            # Calculate the slope from (0, h_val) to the top of the current building k.
            current_building_slope = (H_k - h_val) / X_k
            max_slope_so_far = max(max_slope_so_far, current_building_slope)
        
        # If the loop completes without any building being invisible, all buildings are visible.
        return True

    # First, check if all buildings are visible from height 0.
    # If so, according to the problem statement, output -1.
    if check(0.0):
        print("-1")
        return

    # Perform binary search to find the maximum height 'h' from which not all buildings are visible.
    # 'low' will track the largest 'h' for which 'check(h)' is False.
    # 'high' will track the smallest 'h' for which 'check(h)' is True.
    low = 0.0
    # The upper bound for 'h' can be very large. A safe upper bound derived from max possible
    # (H_j * X_i) / (X_i - X_j) is approximately 10^18. Using 2 * 10^18 for robustness.
    high = 2.0 * (10**18) 
    
    # Run a fixed number of iterations for sufficient floating-point precision.
    # 100 iterations are usually enough for 10^-9 precision over a wide range.
    for _ in range(100): 
        mid = (low + high) / 2.0
        if check(mid):
            # If all buildings are visible from 'mid', then 'mid' is too high to be our answer.
            # The answer must be less than or equal to 'mid'.
            high = mid
        else:
            # If not all buildings are visible from 'mid', then 'mid' is a candidate for the answer
            # or we need to look for a higher 'h'. We want the maximum such 'h'.
            low = mid
        
    # After the binary search, 'low' will converge to the maximum 'h' for which 'check(h)' is False.
    # Print the result formatted to 18 decimal places for sufficient precision.
    print(f"{low:.18f}")

solve()