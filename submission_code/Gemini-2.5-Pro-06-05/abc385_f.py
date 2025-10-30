import sys

def solve():
    """
    Solves the visibility problem using a convex hull trick approach.

    The problem asks for the maximum non-negative height `h` at coordinate 0
    from which not all `N` buildings are visible. A building `i` is not visible
    if the line of sight to its top is blocked by a prior building `k`.
    This translates to finding the maximum `h` such that `h` is less than or
    equal to the y-intercept of a line connecting the tops of two buildings, `T_k` and `T_i`.

    We iterate through buildings, maintaining an upper convex hull of the building
    tops seen so far. For each new building, we find the "tangent" from its top
    to the hull, which gives a candidate for the maximum blocking height.
    This tangent search is done efficiently using binary search on the hull vertices.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N_str = input()
        if not N_str:
            return
        N = int(N_str)
        
        if N <= 1:
            print(-1)
            return

        # Pre-allocate lists for performance
        X = [0] * N
        H = [0] * N
        for i in range(N):
            X[i], H[i] = map(int, input().split())

    except (IOError, ValueError):
        # Handle cases of empty or malformed input
        return

    # hull_indices stores indices of points on the upper convex hull
    hull_indices = []
    
    # max_h_candidate will store max_{i,k<i} h_block(i,k)
    max_h_candidate = -1.0

    # Iterate through each building
    for i in range(N):
        # Query Step: Find the building on the hull that is the "tangent"
        if hull_indices:
            # Binary search to find the index in hull_indices that minimizes the slope
            # of the line connecting that hull point to the current point T_i.
            # This is equivalent to maximizing the y-intercept.
            l, r = 0, len(hull_indices) - 1
            while l < r:
                m = (l + r) // 2
                k1_idx = hull_indices[m]
                k2_idx = hull_indices[m + 1]

                # Compare slope(k1, i) and slope(k2, i) via cross-multiplication
                # to avoid floating point inaccuracies. Python's integers handle large numbers.
                # (H[i]-H[k1])/(X[i]-X[k1]) vs (H[i]-H[k2])/(X[i]-X[k2])
                val1 = (H[i] - H[k1_idx]) * (X[i] - X[k2_idx])
                val2 = (H[i] - H[k2_idx]) * (X[i] - X[k1_idx])

                if val1 < val2:
                    r = m
                else:
                    l = m + 1
            
            best_k = hull_indices[l]

            # Calculate the critical height h_block(i, best_k)
            # h = (H[k]*X[i] - H[i]*X[k]) / (X[i] - X[k])
            h_block = (H[best_k] * X[i] - H[i] * X[best_k]) / (X[i] - X[best_k])
            
            if max_h_candidate < h_block:
                max_h_candidate = h_block

        # Update Step: Add the current building to the hull, maintaining convexity.
        while len(hull_indices) >= 2:
            p1_idx = hull_indices[-2]
            p2_idx = hull_indices[-1]
            
            # Check if T_p1 -> T_p2 -> T_i is a "right turn" or collinear.
            # This means slope(p1, p2) >= slope(p2, i).
            # Use cross-multiplication to compare slopes with integer arithmetic.
            if (H[p2_idx] - H[p1_idx]) * (X[i] - X[p2_idx]) >= \
               (H[i] - H[p2_idx]) * (X[p2_idx] - X[p1_idx]):
                break
            else:
                hull_indices.pop()
        
        hull_indices.append(i)

    # If max_h_candidate is negative, it's possible to see all buildings from h=0.
    if max_h_candidate < 0:
        print(-1)
    else:
        # Otherwise, the answer is the non-negative height found.
        print(f"{max_h_candidate:.20f}")

solve()