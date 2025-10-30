def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except StopIteration:
        return

    # Group items by type:
    # A_list: pull-tab cans (T=0) yield immediate happiness.
    # B_list: regular cans (T=1) yield happiness if an opener is used.
    # C_list: can openers (T=2) provide "capacity" (number of cans that can be opened).
    A_list = []  
    B_list = []  
    C_list = []  
    for _ in range(N):
        t = int(next(it))
        x = int(next(it))
        if t == 0:
            A_list.append(x)
        elif t == 1:
            B_list.append(x)
        else:
            C_list.append(x)
    
    # sort each list in descending order
    A_list.sort(reverse=True)
    B_list.sort(reverse=True)
    C_list.sort(reverse=True)
    
    na = len(A_list)
    nb = len(B_list)
    nc = len(C_list)
    
    # Precompute prefix sums, so PA[i] is the sum of the i highest values in A_list, etc.
    PA = [0]*(na+1)
    for i in range(na):
        PA[i+1] = PA[i] + A_list[i]
    PB = [0]*(nb+1)
    for i in range(nb):
        PB[i+1] = PB[i] + B_list[i]
    PC = [0]*(nc+1)
    for i in range(nc):
        PC[i+1] = PC[i] + C_list[i]
    
    # We will try all possibilities for choosing c items from C.
    # Then the remaining k = M - c items must come from A and B.
    # In order to get happiness from a regular can (from B) you must have an available can opener,
    # and the chosen C items supply capacity = PC[c].
    #
    # If from the k items we take r items from B and (k - r) items from A,
    # the total happiness is f(r) = PA[k - r] + PB[r].
    #
    # There are feasibility conditions:
    #   • k must not exceed (na + nb) because there is no benefit from picking extra A or B if not available.
    #   • We must choose at least r_min = max(0, k - na) items from B (since we cannot take more than na from A).
    #   • We can take at most r_max_raw = min(k, nb) items from B.
    #   • And also r must be ≤ capacity = PC[c].
    #
    # Hence we restrict r to be in [L, R] with:
    #   L = max(0, k - na) and R = min(k, nb, PC[c]).
    # The function f (which is concave) is maximized by a unique r (or at least the maximum is attained at an endpoint).
    # We use a binary‐search style (unimodal search) to find the best r quickly.
    
    ans = 0
    max_c = M if M < nc else nc  # we can't pick more C items than available or than M.
    for c in range(0, max_c+1):
        k = M - c   # number of items to choose from A and B
        if k > na + nb:
            continue  # not enough items from A and B
        cap = PC[c]  # available capacity from chosen can openers
        
        # r must be at least: we need to fill k items from A U B; if A_list doesn't suffice, then r must be >= k - na.
        r_low = 0 if k <= na else k - na  
        # r cannot exceed the number available in B: r_max_raw = min(k, nb)
        r_high_full = k if k < nb else nb
        # Also, we cannot use more B items than we can "open" (capacity).
        r_high = r_high_full if cap > r_high_full else cap
        if r_low > r_high:
            continue  # not feasible because even the minimum r needed exceeds capacity.
        
        # f(r) = PA[k - r] + PB[r]. This function is concave.
        # Use binary search on the interval [r_low, r_high] to find the maximum.
        pA = PA   # local variables for speed
        pB = PB
        lo = r_low
        hi = r_high
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            f_mid = pA[k - mid] + pB[mid]
            f_mid1 = pA[k - (mid+1)] + pB[mid+1]
            if f_mid < f_mid1:
                lo = mid
            else:
                hi = mid
        candidate = pA[k - lo] + pB[lo]
        cand2 = pA[k - hi] + pB[hi]
        best_here = candidate if candidate >= cand2 else cand2
        if best_here > ans:
            ans = best_here
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()