def main():
    import sys
    input_data=sys.stdin.read().strip().split()
    N=int(input_data[0])
    Q=int(input_data[1])
    P=[0]*Q
    V=[0]*Q
    idx=2
    for i in range(Q):
        P[i]=int(input_data[idx]); 
        V[i]=int(input_data[idx+1])
        idx+=2

    MOD=998244353
    
    # ----------------------------------------------------------------
    # If for some pair i<j we have V[i] > V[j] but P[i] == P[j],
    # then there is no way to avoid overlap on that exact position,
    # so answer = 0 immediately.
    # Otherwise, each operation i can be forced to "L" or "R" by
    # comparing P[i] with those P[j] where V[j]>V[i] and j<i (left side constraints)
    # and those P[j] where V[i]>V[j] and j>i (right side constraints).
    #
    # Detailed reasoning:
    # 1) For j<i, if V[j]>V[i], then intervals i and j must be disjoint.
    #    Since j is performed first, we look at j->i:
    #      - If P[i] > P[j], that pair forces j=L and i=R;
    #        viewed from i's perspective, "all such j imply P[j]<P[i]" means i must be R.
    #      - If P[i] < P[j], that pair forces j=R and i=L;
    #        so "all such j imply P[j]>P[i]" means i must be L.
    #    If among those j we find both P[j]<P[i] and P[j]>P[i], it's impossible.
    #
    # 2) For j>i, if V[i]>V[j], then intervals i and j must be disjoint.
    #    Since i is before j, we look at i->j:
    #      - If P[j] > P[i], that forces i=L, j=R;
    #        from i's perspective, "for all j with V[i]>V[j], if P[j]>P[i], i must be L."
    #      - If P[j] < P[i], that forces i=R, j=L;
    #        so "for all j in that set, if P[j]<P[i], i must be R."
    #    Again, if there is a mix of P[j] < P[i] and P[j] > P[i] in that set, no solution.
    #
    # Thus for each i, we gather:
    #   L_i = { j | j < i and V[j] > V[i] }
    #   R_i = { j | j > i and V[i] > V[j] }
    # Check if within L_i all P[j] are < P[i] or all > P[i] or mixed (then 0).
    #   all < => i forced R, all > => i forced L
    # Check R_i similarly:
    #   all > => i forced L, all < => i forced R
    # Combine the two forced directions.  If conflict => 0, else fix i's side or leave free.
    # Finally the number of free i is f.  The result = 2^f mod 998244353.
    #
    # Complexity: O(Q^2), which should be just on the edge but doable with efficient code.
    # ----------------------------------------------------------------

    # We will store for each i:
    #   must_be_left[i] = False/True
    #   must_be_right[i] = False/True
    # Initially all False.
    # If we ever discover a direct contradiction, answer=0.
    # In the end, if must_be_left[i] and must_be_right[i] are both true => 0
    # If only one is true, i is forced. If both are false, i is free.
    
    must_be_left = [False]*Q
    must_be_right = [False]*Q
    
    # First: quick check for (V_i > V_j and P_i == P_j) => 0
    # We'll do it in the same double loop we use for building constraints.
    # But let's separate the logic for clarity (still O(Q^2)).
    
    # We'll gather for each i:
    #   set_L = indices j< i where V[j]>V[i]
    #   we check P[j] relative to P[i].
    #   set_R = indices j> i where V[i]>V[j]
    #   check P[j] relative to P[i].
    # We'll do it in one pass for each i with an inner loop for j.
    
    # We'll define a helper function to see if a group of P_j is all < P_i, all > P_i, or mixed.
    def check_homogeneous_pvals(pvals, pivot):
        # returns:
        #   'all_less' if all < pivot
        #   'all_greater' if all > pivot
        #   'mixed' if there's at least one < pivot and one > pivot
        #   'equal' if any p == pivot
        if not pvals:  # empty
            return 'empty'
        less = False
        greater = False
        for x in pvals:
            if x == pivot:
                return 'equal'
            if x < pivot:
                less = True
            else:
                greater = True
            if less and greater:
                return 'mixed'
        if less and not greater:
            return 'all_less'
        if greater and not less:
            return 'all_greater'
        # Should never get here logically, but just in case:
        return 'mixed'
    
    # We do the double loop, but to avoid scanning each set multiple times we can just
    # collect them for each i on the fly.
    
    for i in range(Q):
        # Collect the sets (actually just the P-values) for L_i and R_i
        left_pvals = []
        right_pvals = []
        
        Vi = V[i]
        Pi = P[i]
        
        # j < i => candidate for left set if V[j] > V[i]
        # j > i => candidate for right set if V[i] > V[j]
        for j in range(i):
            # j < i
            if V[j] > Vi:
                # belongs to L_i
                # also check if P[j] == P[i]
                if P[j] == Pi:
                    # no solution
                    print(0)
                    return
                left_pvals.append(P[j])
        for j in range(i+1, Q):
            # j > i
            if Vi > V[j]:
                # belongs to R_i
                if P[j] == Pi:
                    print(0)
                    return
                right_pvals.append(P[j])
        
        # Now interpret left_pvals
        # if non-empty, must all be < Pi or all > Pi. If mixed => no solution
        # if all < Pi => i forced R, if all > Pi => i forced L
        resL = check_homogeneous_pvals(left_pvals, Pi)
        if resL == 'equal' or resL == 'mixed':
            # no solution
            print(0)
            return
        elif resL == 'all_less':
            # i forced to R
            must_be_right[i] = True
        elif resL == 'all_greater':
            # i forced to L
            must_be_left[i] = True
        # if 'empty', no constraint from L_i
        
        # Now interpret right_pvals
        # if all > Pi => i forced L, if all < Pi => i forced R
        resR = check_homogeneous_pvals(right_pvals, Pi)
        if resR == 'equal' or resR == 'mixed':
            print(0)
            return
        elif resR == 'all_greater':
            # i forced L
            must_be_left[i] = True
        elif resR == 'all_less':
            # i forced R
            must_be_right[i] = True
        # if 'empty', no constraint from R_i
        
        # check if we ended up with a direct contradiction
        if must_be_left[i] and must_be_right[i]:
            # forced both ways => impossible
            print(0)
            return
    
    # Now count how many i are "free" => not forced either way
    # i.e. must_be_left[i] == False and must_be_right[i] == False
    free_count = 0
    for i in range(Q):
        if not must_be_left[i] and not must_be_right[i]:
            free_count += 1
    
    # number of ways = 2^free_count % MOD
    # fast exponent:
    # (we could use pow(base, exp, mod))
    ans = pow(2, free_count, MOD)
    print(ans)