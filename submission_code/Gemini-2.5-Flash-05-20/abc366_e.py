import sys
from bisect import bisect_left, bisect_right

def solve():
    N, D = map(int, sys.stdin.readline().split())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))

    X_coords = sorted([p[0] for p in points])
    Y_coords = sorted([p[1] for p in points])

    P_x = [0] * (N + 1)
    for i in range(N):
        P_x[i+1] = P_x[i] + X_coords[i]

    P_y = [0] * (N + 1)
    for i in range(N):
        P_y[i+1] = P_y[i] + Y_coords[i]

    # Function to calculate sum of absolute differences efficiently
    # S_v(val) = sum_{i=1 to N} |val - v_i|
    # This formula needs idx to be count of elements <= val. Use bisect_right.
    def calc_abs_sum(val, coords, P_coords):
        idx = bisect_right(coords, val)
        # sum_{v_i <= val} (val - v_i) = idx * val - P_coords[idx]
        # sum_{v_i > val} (v_i - val) = (P_coords[N] - P_coords[idx]) - (N - idx) * val
        # Total sum = (idx * val - P_coords[idx]) + (P_coords[N] - P_coords[idx]) - (N - idx) * val
        return (2 * idx - N) * val + P_coords[N] - 2 * P_coords[idx]

    # Max/min bounds for coordinates x, y that can possibly satisfy the condition.
    # x_i, y_i are in [-10^6, 10^6]. D is up to 10^6.
    # The relevant range for x and y is roughly [-2*10^6, 2*10^6].
    MIN_SEARCH_RANGE = -2 * 10**6
    MAX_SEARCH_RANGE = 2 * 10**6

    # Find the valid range for x such that S_x(x) <= D.
    # This range is [x_low_valid, x_high_valid].
    # Initialize with values outside the possible range.
    x_low_valid = MAX_SEARCH_RANGE + 1 
    x_high_valid = MIN_SEARCH_RANGE - 1

    # Binary search for x_low_valid (smallest x)
    low, high = MIN_SEARCH_RANGE, MAX_SEARCH_RANGE
    while low <= high:
        mid = (low + high) // 2
        if calc_abs_sum(mid, X_coords, P_x) <= D:
            x_low_valid = mid
            high = mid - 1
        else:
            low = mid + 1

    # Binary search for x_high_valid (largest x)
    low, high = MIN_SEARCH_RANGE, MAX_SEARCH_RANGE
    while low <= high:
        mid = (low + high) // 2
        if calc_abs_sum(mid, X_coords, P_x) <= D:
            x_high_valid = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # If no x satisfies S_x(x) <= D, then total_count is 0.
    if x_low_valid > x_high_valid:
        print(0)
        return

    total_count = 0

    # Iterate x from x_low_valid to x_high_valid
    for x in range(x_low_valid, x_high_valid + 1):
        Sx_val = calc_abs_sum(x, X_coords, P_x)
        
        # If S_x(x) alone exceeds D, no y can satisfy the condition.
        if Sx_val > D:
            continue

        remaining_D = D - Sx_val

        # Find y_low (smallest y) such that S_y(y) <= remaining_D
        y_low_for_this_x = MAX_SEARCH_RANGE + 1
        low, high = MIN_SEARCH_RANGE, MAX_SEARCH_RANGE
        while low <= high:
            mid = (low + high) // 2
            if calc_abs_sum(mid, Y_coords, P_y) <= remaining_D:
                y_low_for_this_x = mid
                high = mid - 1
            else:
                low = mid + 1

        # Find y_high (largest y) such that S_y(y) <= remaining_D
        y_high_for_this_x = MIN_SEARCH_RANGE - 1
        low, high = MIN_SEARCH_RANGE, MAX_SEARCH_RANGE
        while low <= high:
            mid = (low + high) // 2
            if calc_abs_sum(mid, Y_coords, P_y) <= remaining_D:
                y_high_for_this_x = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Add the count of valid y values for this x
        if y_low_for_this_x <= y_high_for_this_x:
            total_count += (y_high_for_this_x - y_low_for_this_x + 1)

    print(total_count)