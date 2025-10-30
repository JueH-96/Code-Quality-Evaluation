def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    D = int(input_data[1])
    coords = list(map(int, input_data[2:]))

    # Separate x and y coordinates
    x_list = coords[0::2]
    y_list = coords[1::2]

    # Quick edge-case: if D=0 and N>1 then answer is 0 immediately 
    # because all points are distinct -> sum of distances > 0 for any (x,y).
    # We can check, but let's allow the general logic to handle that.
    # (It will yield 0 anyway.)

    # Sum, min, max of x and y
    sum_x = sum(x_list)
    sum_y = sum(y_list)
    minx = min(x_list)
    maxx = max(x_list)
    miny = min(y_list)
    maxy = max(y_list)

    # A small helper for ceiling of (a / b) where b>0.
    # Example: ceil_div(-3, 2) = -1, ceil_div(3,2)=2, etc.
    def ceil_div(a, b):
        return -((-a)//b)

    # Compute candidate bounds for x
    # We want all x that might possibly satisfy sum_i |x - x_i| <= D.
    # For x < minx, the sum is sum_i (x_i - x) = sum_x - N*x, which must be <= D => x >= (sum_x - D)/N
    # Similarly for x > maxx.  So we combine those with the actual minx, maxx.
    x_min_candidate_1 = (sum_x - D)//N     # floor((sum_x - D)/N)
    x_max_candidate_1 = ceil_div(sum_x + D, N)  # ceil((sum_x + D)/N)
    X_min = min(x_min_candidate_1, minx) - 2  # a small margin
    X_max = max(x_max_candidate_1, maxx) + 2

    # Build a frequency array for x-coordinates in [X_min .. X_max]
    X_range = X_max - X_min + 1
    if X_range <= 0:
        # No feasible x at all
        print(0)
        return
    x_count = [0]*(X_range)
    # Offset to map x -> x - X_min
    for x in x_list:
        idx = x - X_min
        x_count[idx] += 1

    # Build f_val[x - X_min] = sum_i |x - x_i| for x in [X_min..X_max].
    # We'll do a linear sweep:
    f_val = [0]*X_range
    base_x = X_min
    # Compute f_val[0] in O(N)
    s = 0
    for xi in x_list:
        s += abs(xi - base_x)
    f_val[0] = s

    # p = # of x_i <= base_x
    p = x_count[0]
    # Fill the rest using the known recurrence:
    # f_val[x+1] = f_val[x] + (2 * (# of x_i <= x+1) - N).
    # We'll keep track of p' = # of x_i <= x+1
    for i in range(X_range-1):
        # # of x_i <= (X_min + i + 1) is p + x_count[i+1]
        p_prime = p + x_count[i+1]
        f_val[i+1] = f_val[i] + (2*p_prime - N)
        p = p_prime

    # Do the same for y
    y_min_candidate_1 = (sum_y - D)//N
    y_max_candidate_1 = ceil_div(sum_y + D, N)
    Y_min = min(y_min_candidate_1, miny) - 2
    Y_max = max(y_max_candidate_1, maxy) + 2
    Y_range = Y_max - Y_min + 1
    if Y_range <= 0:
        # No feasible y at all
        print(0)
        return
    y_count = [0]*Y_range
    for y in y_list:
        idx = y - Y_min
        y_count[idx] += 1

    G_val = [0]*Y_range
    base_y = Y_min
    s = 0
    for yi in y_list:
        s += abs(yi - base_y)
    G_val[0] = s

    q = y_count[0]
    for i in range(Y_range-1):
        q_prime = q + y_count[i+1]
        G_val[i+1] = G_val[i] + (2*q_prime - N)
        q = q_prime

    # Now we only need to quickly answer "how many y in [Y_min..Y_max] have G_val <= T?"
    # for T = 0..D (or T> D). We'll build a small frequency array freq_array[0..D],
    # plus freq_over for all values > D.
    freq_array = [0]*(D+1)
    freq_over = 0
    for valG in G_val:
        if valG <= D:
            freq_array[valG] += 1
        else:
            freq_over += 1

    # Build prefix sum ps so ps[t] = # of y with G_val <= t, for t in [0..D].
    ps = [0]*(D+1)
    ps[0] = freq_array[0]
    for i in range(1, D+1):
        ps[i] = ps[i-1] + freq_array[i]

    # A helper to get number of y where G_val <= T
    def y_count_leq(T):
        if T < 0:
            return 0
        if T <= D:
            return ps[T]
        # T > D => all y
        return ps[D] + freq_over

    # Finally, sum over x in [X_min..X_max] of the number of valid y
    # valid y means G_val <= (D - f_val[x - X_min]) if that is >=0
    result = 0
    for i in range(X_range):
        fx = f_val[i]
        if fx <= D:
            T = D - fx  # how many y have G_val <= T
            result += y_count_leq(T)

    print(result)

# Remember to call main()!
if __name__ == "__main__":
    main()