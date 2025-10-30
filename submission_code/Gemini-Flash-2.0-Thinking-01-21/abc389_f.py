import sys
from bisect import bisect_right

# Increase recursion depth for large inputs if needed
# sys.setrecursionlimit(200005 * 2) # Not typically needed for this logic

def solve():
    # Increase input reading speed
    input = sys.stdin.readline

    N = int(input())
    contests = []
    for _ in range(N):
        l, r = map(int, input().split())
        contests.append((l, r))

    MAX_VAL = 500000 # Max initial rating and R_i
    # pts stores the sorted critical points.
    # delta[i] is the increment for the interval [pts[i], pts[i+1]-1]
    # The domain is [1, MAX_VAL] for initial ratings.
    # The critical points track the function g_k(r) = r + delta(r) for r in [1, MAX_VAL].
    pts = [1, MAX_VAL + 1]
    delta = [0]

    for L_k, R_k in contests:
        new_points_set = set() # Use set for unique points

        # Iterate through current intervals [a, b] = [pts[j], pts[j+1]-1]
        for j in range(len(pts) - 1):
            a = pts[j]
            b = pts[j+1] - 1
            inc = delta[j]

            # The values g_{k-1}(r) for r in [a, b] are in [a+inc, b+inc].
            # Condition for +1 increment: L_k <= g_{k-1}(r) <= R_k
            # <=> L_k <= r + inc <= R_k
            # <=> L_k - inc <= r <= R_k - inc
            
            # Find the intersection of [a, b] and [L_k - inc, R_k - inc]
            # This is the range [ris, rie] where r in [a, b] gets +1 increment
            ris = max(a, L_k - inc)
            rie = min(b, R_k - inc)

            if ris <= rie:
                # The range [ris, rie] within [a, b] gets +1 increment.
                # This creates new boundaries at ris and rie+1 relative to [a, b].
                # These become new critical points in pts_new if they split the interval (a, b+1).
                # ris splits if a < ris <= b.
                # rie+1 splits if a <= rie < b.
                
                # Add points that are strictly inside the interval (a, b+1).
                # ris is a critical point if it's > a.
                # rie + 1 is a critical point if it's <= b.
                # Check boundary conditions carefully.
                # The points where the increment function potentially changes are the boundaries of the [ris, rie] interval.
                # These points are `ris` and `rie + 1`.
                # We add `ris` as a critical point if it falls strictly after `a`.
                if ris > a:
                    new_points_set.add(ris)
                # We add `rie + 1` as a critical point if it falls strictly before `b + 1`.
                if rie < b: # This implies rie+1 <= b. Check if rie+1 is before the next boundary b+1.
                     new_points_set.add(rie + 1)


        # Add all existing points to the new set. These points are always critical boundaries.
        for p in pts:
            new_points_set.add(p)

        pts_new = sorted(list(new_points_set))
        delta_new = []

        # Calculate new delta for each interval [p'_i, p'_{i+1}-1] in pts_new
        # Use two pointers to efficiently find old delta
        ptr_old_pts = 0 # Pointer for the old pts list

        for i in range(len(pts_new) - 1):
            r = pts_new[i] # Start of the new interval [r, pts_new[i+1]-1]

            # Advance ptr_old_pts to find the interval [pts[ptr_old_pts], pts[ptr_old_pts+1]-1] containing r
            # The pointer should point to the index `j` such that `pts[j] <= r < pts[j+1]`.
            # Since `pts_new` contains all points from `pts`, `r` is either a point in `pts` or a new point.
            # If `r` is in `pts`, say `r = pts[k]`, then the interval is `[pts[k], pts[k+1]-1]`, index is `k`.
            # If `r` is a new point added strictly between `pts[j]` and `pts[j+1]`, it falls in `[pts[j], pts[j+1]-1]`, index is `j`.
            # The `while` loop correctly finds the index `j` such that `pts[j] <= r < pts[j+1]`.
            while ptr_old_pts < len(pts) - 1 and pts[ptr_old_pts + 1] <= r:
                ptr_old_pts += 1
            old_inc = delta[ptr_old_pts]

            # Check if this new interval [r, pts_new[i+1]-1] falls into the range that gets +1 increment.
            # The range that gets +1 increment for the original interval [pts[ptr_old_pts], pts[ptr_old_pts+1]-1]
            # with old_inc is [ris_val, rie_val] where ris_val = L_k - old_inc and rie_val = R_k - old_inc.
            # Since pts_new includes all critical points, the new interval [r, pts_new[i+1]-1]
            # must lie entirely within [a, ris-1], [ris, rie], or [rie+1, b] regions of the old interval [a, b].
            # We just need to check which region `r` falls into relative to `[ris_val, rie_val]`.
            # If `ris_val <= r <= rie_val`, the new increment for [r, pts_new[i+1]-1] is old_inc + 1.
            # Otherwise, it's old_inc.

            ris_val = L_k - old_inc
            rie_val = R_k - old_inc

            if ris_val <= r <= rie_val:
                 delta_new.append(old_inc + 1)
            else:
                 delta_new.append(old_inc)


        pts = pts_new
        delta = delta_new

    # Process Queries
    Q = int(input())
    results = []
    for _ in range(Q):
        X = int(input())
        # Find the interval [pts[j], pts[j+1]-1] that contains X
        # bisect_right(pts, X) returns k such that pts[k-1] <= X < pts[k]
        # We need index j such that pts[j] <= X < pts[j+1]. This is k-1.
        j = bisect_right(pts, X) - 1
        final_rating = X + delta[j]
        results.append(final_rating)

    for res in results:
        print(res)

solve()