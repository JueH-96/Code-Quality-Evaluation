# YOUR CODE HERE
def main():
    import sys, math
    import numpy as np
    # Read entire input from sys.stdin.buffer (this is fast)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # First two numbers: N and M.
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    # Next N integers are the P_i values.
    P_list = [int(next(it)) for _ in range(N)]
    
    # Convert the list of prices to a numpy array (using 64-bit int)
    P_arr = np.array(P_list, dtype=np.int64)
    # Also create a long-double array (np.longdouble) for cost computations
    P_ld = P_arr.astype(np.longdouble)
    # For the inner binary search (see below) we need the smallest price.
    minP = int(P_arr.min())
    
    # For a given threshold t, we “simulate” buying all increments (units)
    # whose marginal cost is <= t.
    # For product i, the number of units purchased is:
    #    k_i = ((t // P_i) + 1) // 2
    # (because the condition P_i*(2*k-1) <= t rearranges to k <= (t/P_i+1)/2)
    # and the cost incurred for that product is P_i*k_i².
    def get_count_and_cost(t):
        t_val = np.int64(t)
        # Vectorized computation: for every product i, compute k_i = ((t//P_i)+1)//2
        k = ((t_val // P_arr) + 1) // 2
        cnt = int(k.sum())
        # Compute cost = sum P_i * (k_i²).
        # To avoid 64-bit overflow (since k_i can be large) we cast k to np.longdouble.
        k_ld = k.astype(np.longdouble)
        cost = np.sum(P_ld * (k_ld * k_ld))
        return cnt, cost

    # Given a candidate total unit count x, we want to compute the minimal cost
    # required to buy x units if we choose the x cheapest increments.
    # “Idea:” We binary search for the smallest t (threshold) so that f(t) >= x.
    # Then the minimal cost is: cost(x) = g(t) - (f(t)-x)*t.
    def cost_for_x(x):
        if x <= 0:
            return 0
        # We know that if we “buy only from the cheapest product” (price = minP),
        # to buy x units you need a threshold t at least minP*(2*x - 1).
        # So we choose hi_t = minP*(2*x) as an upper bound.
        lo_t = 0
        hi_t = minP * (2 * x)
        while lo_t < hi_t:
            mid = (lo_t + hi_t) // 2
            cnt, _ = get_count_and_cost(mid)
            if cnt >= x:
                hi_t = mid
            else:
                lo_t = mid + 1
        t_found = lo_t
        cnt, tot_cost = get_count_and_cost(t_found)
        surplus = cnt - x
        minimal_cost = tot_cost - surplus * np.longdouble(t_found)
        return minimal_cost

    # Now, we want to choose the maximum total unit count x such that cost_for_x(x) <= M.
    # An upper bound for x: if each product is used separately, the maximum units
    # from product i would be floor(sqrt(M / P_i)). Hence an upper bound is:
    #    x_upper = sum_{i=1}^N floor( sqrt(M / P_i) )
    # (We use np.longdouble for a safe square-root computation.)
    P_ld_forSqrt = P_ld
    sqrt_M = np.longdouble(M) ** 0.5
    sqrt_div = sqrt_M / np.sqrt(P_ld_forSqrt)
    x_upper = int(np.sum(np.floor(sqrt_div)))
    
    # Perform a binary search on x.
    lo_x = 0
    hi_x = x_upper
    while lo_x < hi_x:
        mid_x = (lo_x + hi_x + 1) // 2
        if cost_for_x(mid_x) <= M:
            lo_x = mid_x
        else:
            hi_x = mid_x - 1
    sys.stdout.write(str(lo_x))

if __name__ == '__main__':
    main()