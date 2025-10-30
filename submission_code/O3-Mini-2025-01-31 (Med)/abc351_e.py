def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # The allowed moves are diagonal.
    # A rabbit can only reach from A=(x1,y1) to B=(x2,y2) if (x1+y1) and (x2+y2)
    # have the same parity. If they differ, answer is defined as 0.
    #
    # It is useful to transform coordinates:
    #   U = x + y,   V = x - y.
    # Then the allowed moves become:
    #   (x+1,y+1)  -> (U+2, V)  
    #   (x+1,y-1)  -> (U, V+2)
    #   (x-1,y+1)  -> (U, V-2)
    #   (x-1,y-1)  -> (U-2, V)
    # Notice that only one of U or V changes (by 2) in a jump.
    #
    # Thus, when going from A to B:
    #   The required changes are: ΔU = (x2+y2) - (x1+y1),  ΔV = (x2-y2) - (x1-y1)
    # Each jump can only change one of these by 2. Hence the minimal number
    # of steps required is exactly:
    #   (|ΔU|/2 + |ΔV|/2) = (|ΔU| + |ΔV|) // 2.
    # (Note: ΔU and ΔV are always even when the point is reachable.)
    #
    # We need to sum this distance over all pairs i<j.
    # But observe: if two points are not reachable (their (x+y) parity differs),
    # their distance is defined as 0.
    #
    # Therefore, we may split the points into two groups according to the parity of (x+y)
    # Let for a group we define for each point:
    #   U = x+y,   V = x-y.
    # Then for any pair in the same group their distance is:
    #   (|U_i - U_j| + |V_i - V_j|) // 2.
    #
    # Our task reduces to computing, for each group, the sum:
    #   Sum_{i < j} (|U_i-U_j| + |V_i-V_j|),
    # then divide by 2.
    #
    # Note: The summations over all pairs (i<j) can be computed efficiently
    # using sorting and using the formula:
    #   Sum_{i<j} |a_i - a_j| = ∑_{i=0}^{k-1} a[i]*(2*i - k + 1)
    # after sorting array a.
    
    # Separate points into groups based on parity of (x+y).
    groups = {0: [], 1: []}  # Each group will hold tuples (U, V).
    for _ in range(n):
        x = int(next(it))
        y = int(next(it))
        U = x + y
        V = x - y
        # Use (x+y) mod 2 as the group key.
        groups[U & 1].append((U,V))
    
    # Helper function to compute sum of differences over all pairs for a list.
    def sum_abs_diff(arr):
        arr.sort()
        k = len(arr)
        total = 0
        for i, a in enumerate(arr):
            total += a * (2 * i - k + 1)
        return total

    total_sum = 0
    # Process each group separately.
    for key in groups:
        pts = groups[key]
        count = len(pts)
        if count < 2:
            continue
        # Separate U and V values.
        U_list = [p[0] for p in pts]
        V_list = [p[1] for p in pts]
        # Sum of |U_i - U_j| for all pairs
        sumU = sum_abs_diff(U_list)
        # Sum of |V_i - V_j| for all pairs
        sumV = sum_abs_diff(V_list)
        # For every pair in the group, contribution is (|ΔU| + |ΔV|) // 2
        # So, the sum for this group is (sumU + sumV) // 2.
        total_sum += (sumU + sumV) // 2
    
    sys.stdout.write(str(total_sum))


if __name__ == '__main__':
    main()