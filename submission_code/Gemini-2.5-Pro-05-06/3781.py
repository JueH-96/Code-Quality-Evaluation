import math

class Solution:
  def maxDistance(self, side: int, points: list[list[int]], k: int) -> int:

    # Helper to calculate Manhattan distance
    def manhattan_distance(p1_coords, p2_coords):
        return abs(p1_coords[0] - p2_coords[0]) + abs(p1_coords[1] - p2_coords[1])

    # Map points to 1D perimeter coordinates and sort them
    # The perimeter coordinate 'p' increases counter-clockwise from (0,0).
    # (0,0) -> p=0
    # (x,0) on bottom edge -> p=x
    # (side,0) -> p=side
    # (side,y) on right edge -> p=side+y
    # (side,side) -> p=2*side
    # (x,side) on top edge (x decreases from side to 0) -> p=2*side + (side-x)
    # (0,side) -> p=3*side
    # (0,y) on left edge (y decreases from side to 0) -> p=3*side + (side-y)
    # (0,0) would be p=4*side if completing the loop this way.
    
    mapped_points = []
    for i, (x, y) in enumerate(points):
        p_val = -1
        # Order of checks ensures corners are handled by a specific edge definition.
        if y == 0:  # Bottom edge (includes (0,0) and (side,0))
            p_val = x
        elif x == side:  # Right edge (includes (side,side) but not (side,0))
            p_val = side + y
        elif y == side:  # Top edge (includes (0,side) but not (side,side))
            p_val = 2 * side + (side - x)
        elif x == 0:  # Left edge (includes neither (0,0) nor (0,side) if y conditions were y>0 and y<side)
                      # Current conditions make it cover (0,y) for any y not yet covered.
                      # Effectively, (0,y) for y decreasing from (not including) side to (not including) 0.
                      # (0,0) is p=0 via y==0 rule. (0,side) is p=3*side via y==side rule.
            p_val = 3 * side + (side - y)
        
        mapped_points.append({'p': p_val, 'x': x, 'y': y})

    # Sort points by perimeter coordinate
    mapped_points.sort(key=lambda pt: pt['p'])
    
    n = len(points)
    
    # sorted_P stores just the (x,y) coordinates, sorted by perimeter
    sorted_P = [(pt['x'], pt['y']) for pt in mapped_points]

    # Memoization for can_achieve results for different d values
    # This can be helpful if binary search explores values that might repeat
    # or if values of d are constrained (not the case here, d is continuous).
    memo_can_achieve_results = {} 

    # Decision function: can we select k points with min_dist >= d_target?
    def can_achieve(d_target):
        if d_target in memo_can_achieve_results:
            return memo_can_achieve_results[d_target]

        # dp[c][i] stores the index (in sorted_P) of the *first* point
        # in a valid chain of 'c' points that ends with sorted_P[i].
        # A value of -1 indicates no such chain exists.
        # 'c' (1-indexed) is the number of points selected in the chain, from 1 to k.
        
        # dp[count_selected][idx_of_last_point_in_chain]
        dp = [[-1] * n for _ in range(k + 1)]

        # Base case: selecting 1 point.
        # Any point sorted_P[i] can be the first point in a chain of length 1.
        # The chain is just sorted_P[i] itself. So, it starts at index i.
        for i in range(n):
            dp[1][i] = i
        
        # Fill DP table
        # num_pts_in_chain = number of points in the current chain being built
        for num_pts_in_chain in range(2, k + 1):
            # i = index of the current point sorted_P[i],
            # which we are trying to make the last point of the chain.
            for i in range(n):
                # prev_chain_last_idx = index of sorted_P[j],
                # which was the last point of a chain of (num_pts_in_chain-1) points.
                # We require j < i so that points are chosen in increasing order of perimeter coordinate.
                for prev_chain_last_idx in range(i): 
                    # If a chain of (num_pts_in_chain-1) ending at sorted_P[prev_chain_last_idx] exists
                    if dp[num_pts_in_chain - 1][prev_chain_last_idx] != -1:
                        # This is the index of the first point in the chain of (num_pts_in_chain-1) points
                        first_point_idx_in_prev_chain = dp[num_pts_in_chain - 1][prev_chain_last_idx]
                        
                        # Current point sorted_P[i] must be far from sorted_P[prev_chain_last_idx]
                        if manhattan_distance(sorted_P[i], sorted_P[prev_chain_last_idx]) >= d_target:
                            # If sorted_P[i] can extend this chain:
                            # The new chain of length 'num_pts_in_chain' ends at sorted_P[i],
                            # and it starts with 'first_point_idx_in_prev_chain'.
                            dp[num_pts_in_chain][i] = first_point_idx_in_prev_chain
                            # Optimization: Found a way to form this chain. No need to check other prev_chain_last_idx.
                            break 

        # Check for a valid circular selection
        # Iterate through all points sorted_P[i] that could be the k-th point in a chain.
        for i in range(n):
            if dp[k][i] != -1: # If a chain of k points ending at sorted_P[i] exists
                first_point_idx_in_chain = dp[k][i]
                # Check circular condition: last point sorted_P[i] must be far from
                # first point sorted_P[first_point_idx_in_chain].
                if manhattan_distance(sorted_P[i], sorted_P[first_point_idx_in_chain]) >= d_target:
                    memo_can_achieve_results[d_target] = True
                    return True
        
        memo_can_achieve_results[d_target] = False
        return False

    # Binary search for the maximum d
    low = 0
    # Max possible Manhattan distance is between (0,0) and (side,side) = 2*side.
    high = 2 * side  
    ans = 0

    while low <= high:
        mid = (low + high) // 2
        # if mid == 0, can_achieve(0) will be true if n >= k.
        # This is handled correctly by the DP logic.
        if can_achieve(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return ans