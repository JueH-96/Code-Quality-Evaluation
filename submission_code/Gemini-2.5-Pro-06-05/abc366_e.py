# YOUR CODE HERE
import sys
from bisect import bisect_right

def solve():
    """
    This function solves the problem by separating the variables x and y.
    The condition is Sum(|x-xi| + |y-yi|) <= D, which can be rewritten as
    f(x) + g(y) <= D, where f(x) = Sum(|x-xi|) and g(y) = Sum(|y-yi|).

    The main idea is to compute the number of solutions by summing over all possible
    integer values that f(x) can take. Let's say f(x) = d_x. We need to find the
    number of y's such that g(y) <= D - d_x.

    The overall formula for the total count is:
    Total = Sum_{d_x = 0 to D} (Number of x's with f(x)=d_x) * (Number of y's with g(y) <= D-d_x)
    """
    try:
        input = sys.stdin.readline
    except:
        input = lambda: ""

    N_str, D_str = input().split()
    N, D = int(N_str), int(D_str)
    
    points = [tuple(map(int, input().split())) for _ in range(N)]

    X = sorted([p[0] for p in points])
    Y = sorted([p[1] for p in points])

    def compute_counts(coords, max_dist):
        n = len(coords)
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i+1] = pref[i] + coords[i]

        def get_dist(val):
            pos = bisect_right(coords, val)
            res = (pos * val - pref[pos]) + ((pref[n] - pref[pos]) - (n - pos) * val)
            return res

        counts = [0] * (max_dist + 1)

        # Median part
        med_lo_idx = (n - 1) // 2
        med_hi_idx = n // 2
        med_lo_val = coords[med_lo_idx]
        med_hi_val = coords[med_hi_idx]
        
        min_d = get_dist(med_lo_val)
        if min_d <= max_dist:
            counts[min_d] = med_hi_val - med_lo_val + 1

        # Right side walk
        curr_val = med_hi_val
        curr_d = min_d
        coord_idx = med_hi_idx
        
        while True:
            slope = 2 * coord_idx - n
            if coord_idx < n:
                next_val = coords[coord_idx]
            else:
                if slope > 0:
                    while True:
                        curr_d += slope
                        if curr_d > max_dist: break
                        counts[curr_d] += 1
                break

            num_steps = next_val - curr_val
            if slope != 0:
                for _ in range(num_steps):
                    curr_d += slope
                    if curr_d > max_dist: break
                    counts[curr_d] += 1
            if curr_d > max_dist: break
            
            curr_val = next_val
            
            while coord_idx < n and coords[coord_idx] == curr_val:
                coord_idx += 1
            
        # Left side walk
        curr_val = med_lo_val
        curr_d = min_d
        coord_idx = med_lo_idx
        
        while True:
            slope = n - 2 * coord_idx
            if coord_idx > 0:
                next_val = coords[coord_idx - 1]
            else:
                if slope > 0:
                    while True:
                        curr_d += slope
                        if curr_d > max_dist: break
                        counts[curr_d] += 1
                break
            
            num_steps = curr_val - next_val
            if slope != 0:
                for _ in range(num_steps):
                    curr_d += slope
                    if curr_d > max_dist: break
                    counts[curr_d] += 1
            if curr_d > max_dist: break

            curr_val = next_val

            while coord_idx > 0 and coords[coord_idx-1] == curr_val:
                coord_idx -= 1
            coord_idx -= 1
            
        return counts

    counts_x = compute_counts(X, D)
    counts_y = compute_counts(Y, D)

    cum_counts_y = [0] * (D + 1)
    if D >= 0:
        cum_counts_y[0] = counts_y[0]
        for i in range(1, D + 1):
            cum_counts_y[i] = cum_counts_y[i-1] + counts_y[i]

    total_pairs = 0
    for d_x in range(D + 1):
        if counts_x[d_x] > 0:
            d_y_max = D - d_x
            if d_y_max >= 0:
                num_y = cum_counts_y[d_y_max]
                total_pairs += counts_x[d_x] * num_y

    print(total_pairs)

solve()