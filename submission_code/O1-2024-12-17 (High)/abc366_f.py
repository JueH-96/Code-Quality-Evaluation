def main():
    import sys
    from functools import cmp_to_key
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    AB = [(int(input_data[2+2*i]), int(input_data[3+2*i])) for i in range(N)]
    
    # If K=1, we can simply pick the function that maximizes A_i*1 + B_i.
    # But the general DP will also handle K=1 correctly, so no special case needed.
    
    # Comparator for sorting:
    # We say f_i < f_j (i goes before j) if f_i(f_j(1)) >= f_j(f_i(1)).
    # f_i(f_j(1)) = A_i*(A_j + B_j) + B_i = A_i*A_j + A_i*B_j + B_i
    # f_j(f_i(1)) = A_j*(A_i + B_i) + B_j = A_j*A_i + A_j*B_i + B_j
    # difference = [A_i*A_j + A_i*B_j + B_i] - [A_j*A_i + A_j*B_i + B_j]
    #            = A_i*B_j + B_i - A_j*B_i - B_j
    #            = B_j*(A_i - 1) - B_i*(A_j - 1)
    # if diff > 0 => i < j, if diff < 0 => j < i
    def cmp(f1, f2):
        (A1, B1) = f1
        (A2, B2) = f2
        diff = B2*(A1 - 1) - B1*(A2 - 1)
        if diff > 0:
            return -1   # f1 < f2
        elif diff < 0:
            return 1    # f1 > f2
        else:
            # Tie-break (not very critical).  Just keep them as-is or compare A1, A2
            return 0
    
    AB.sort(key=cmp_to_key(cmp))
    
    # We will maintain dp[j] as a list of (a, b) pairs representing
    # distinct (undominated) partial compositions that use exactly j functions.
    # The composition c_j(x) = a*x + b if we have picked j functions so far
    # in the sorted order (left-to-right means these functions are outer→...→inner).
    #
    # Transition:
    # If dp[j] contains (a, b), and we pick next function with (A, B),
    # the new composition c_{j+1}(x) = c_j( f(x) ) where f(x)= A*x + B.
    # So c_{j+1}(x) = a * (A*x + B) + b = (a*A) x + (a*B + b).
    # We store (a2, b2) = (a*A, a*B + b) in dp[j+1].
    #
    # To keep dp[j] small, we do two types of pruning:
    # 1) Bounding prune: if the maximum possible final value using up to (K-j) more picks
    #    cannot exceed our global best_result, we discard the state.
    # 2) Dominance prune: if (a1, b1) and (a2, b2) satisfy a1 <= a2 and b1 <= b2,
    #    then (a1, b1) is dominated and can be removed.
    
    # Maximum possible extension if we always pick (A=50,B=50) for r steps:
    def upper_bound(a_, b_, r_):
        a, b = a_, b_
        for _ in range(r_):
            # next composition step = c_{next}(x) = c(x) with A=50,B=50 inside
            # but in "outside( inside(x) )" form?  Actually we want c( f50(x) ).
            # The formula is c_new(x) = a*(50*x + 50) + b => (a*50)x + (a*50 + b).
            # But we are building from left→right in final expression means
            # c_{next}(x) = c_{current}( f50(x) ) => a_next = a_current*50; b_next = a_current*50 + b_current
            # This matches the lines below:
            new_a = a * 50
            new_b = a * 50 + b
            a, b = new_a, new_b
        return a + b
    
    # Merge two chains (both lists of (a,b) in descending order of a):
    # We'll combine, sort again by descending a, then remove dominated states.
    def merge_chains(chain1, chain2):
        if not chain2:
            return chain1
        merged = chain1 + chain2
        # Sort by descending 'a'. If tie in 'a', higher 'b' first (though not crucial)
        merged.sort(key=lambda x: (x[0], x[1]), reverse=True)
        res = []
        max_b_so_far = -1
        for (a, b) in merged:
            if b > max_b_so_far:
                res.append((a, b))
                max_b_so_far = b
        return res
    
    # Initialize DP structures
    dp = [[] for _ in range(K+1)]
    # dp[j] will store a descending-a chain of (a,b)
    dp[0] = [(1, 0)]  # The "empty composition" c(x)= x => a=1,b=0
    best_result = 0
    
    # Main DP loop over all functions in sorted order
    for (A, B) in AB:
        # We'll build new states for each j in [K-1..0]
        # so we don't overwrite dp[j] before using it
        for j in range(K-1, -1, -1):
            if not dp[j]:
                continue
            # Expand each state in dp[j]
            for (a, b) in dp[j]:
                a2 = a * A
                b2 = a * B + b
                if j+1 == K:
                    # If this is our K-th pick, evaluate final
                    val = a2 + b2
                    if val > best_result:
                        best_result = val
                else:
                    # Otherwise, check bounding
                    ub = upper_bound(a2, b2, K-(j+1))
                    if ub > best_result:
                        # We only store if it has potential
                        dp[j+1] = merge_chains(dp[j+1], [(a2, b2)])
    
    print(best_result)

# Call main() to comply with the requirement.
if __name__ == "__main__":
    main()