import sys

def solve():
    N = int(sys.stdin.readline())
    points = []
    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        points.append((X, Y))

    points_s0 = [] # Group where (X+Y) % 2 == 0
    points_s1 = [] # Group where (X+Y) % 2 == 1

    for X, Y in points:
        if (X + Y) % 2 == 0:
            points_s0.append((X, Y))
        else:
            points_s1.append((X, Y))
            
    total_dist_sum = 0
    
    # Iterate over the two groups of points
    for current_points_group in [points_s0, points_s1]:
        if len(current_points_group) < 2: # Need at least two points to form a pair
            continue

        M = len(current_points_group)
        
        # Calculate u_k = X_k + Y_k and v_k = X_k - Y_k for points in this group
        u_coords = [] 
        v_coords = [] 
        
        for X, Y in current_points_group:
            u_coords.append(X + Y)
            v_coords.append(X - Y)
            
        # Sort u_coords and v_coords to calculate sum of absolute differences
        u_coords.sort()
        v_coords.sort()
        
        # Calculate sum_{i<j} |arr_j - arr_i| for a sorted array arr.
        # This sum is equal to sum_{k=0}^{M-1} (2*k - M + 1) * arr[k].
        
        current_sum_u_diffs = 0
        for k in range(M):
            # Coefficient for arr[k] is k (num elements smaller) - (M-1-k) (num elements larger)
            # = k - M + 1 + k = 2*k - M + 1
            current_sum_u_diffs += (2 * k - M + 1) * u_coords[k]
            
        current_sum_v_diffs = 0
        for k in range(M):
            current_sum_v_diffs += (2 * k - M + 1) * v_coords[k]
            
        # For any two points P_i, P_j in the same group,
        # (u_j-u_i) is even, and (v_j-v_i) is even.
        # (because (X+Y) parities are same, so (X+Y) diff is even;
        #  (X-Y) parities are same ((X-Y)%2 == (X+Y)%2), so (X-Y) diff is even)
        # Therefore, |u_j-u_i| and |v_j-v_i| are even.
        # So, current_sum_u_diffs and current_sum_v_diffs are sums of even numbers,
        # hence they are themselves even.
        # Their sum (current_sum_u_diffs + current_sum_v_diffs) is also even.
        # Thus, integer division // 2 is appropriate and exact.
        
        total_dist_sum += (current_sum_u_diffs + current_sum_v_diffs) // 2
        
    print(total_dist_sum)

solve()