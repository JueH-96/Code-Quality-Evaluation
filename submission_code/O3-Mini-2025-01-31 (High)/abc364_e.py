def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    dishes = []
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        dishes.append((a, b))
        
    # Explanation:
    # Snuke will eat dishes in order. He always “eats” the first dish no matter what;
    # after each dish he checks the new total sweetness and saltiness. If at any point 
    # the cumulative sweetness (or saltiness) exceeds X (or Y) he stops eating any further.
    # Thus if we can arrange an ordering so that the FIRST (k-1) dishes have cumulative
    # sweetness <= X and saltiness <= Y and then add any extra dish (which may “break” the limit)
    # as the kth dish, Snuke will eat k dishes.
    #
    # In other words, if we can “safely” schedule a certain number of dishes (the ones eaten
    # BEFORE the final dish) then we can always append one more dish (even if its addition makes 
    # the total exceed X or Y), and Snuke will count that last dish.
    #
    # Note that if a dish is “unsafe” to be in the safe prefix then it must be used as the extra 
    # (final) dish – but if it is used first, it is allowed (even if it exceeds thresholds), but then 
    # Snuke stops immediately.
    #
    # So one may compute:
    #   m = maximum number of dishes Takahashi can arrange as a prefix so that when eaten in order,
    #       the cumulative sweetness and saltiness never exceed X and Y.
    # Then if there is at least one dish that is not used in that safe prefix (i.e. m < N)
    # we may get m (safe prefix) + 1 (the extra dish that triggers the “stop”) eaten in total.
    # Otherwise if m == N, then all dishes can be eaten safely.
    #
    # How do we compute m? Notice that when dishes have strictly positive sweetness and saltiness,
    # if the total sum of a list of dishes is <= (X,Y), then in any order the cumulative (prefix) sums 
    # will be at most the final total – so an ordering exists.
    #
    # But a dish can start the safe prefix only if it is individually safe (i.e. a <= X and b <= Y)
    # For the safe prefix we consider only the dishes which are individually safe.
    safe = []
    for (a, b) in dishes:
        if a <= X and b <= Y:
            safe.append((a,b))
            
    # We now do a knapSack‐like DP:
    # Let dp[(s,t)] = maximum number of safe dishes (from those that are individually safe)
    # whose totals add up to (s,t) with s <= X, t <= Y
    dp = {(0, 0): 0}
    for (a,b) in safe:
        new_dp = dict(dp)
        for (s, t), cnt in dp.items():
            ns = s + a
            nt = t + b
            if ns <= X and nt <= Y:
                new_cnt = cnt + 1
                key = (ns, nt)
                if key not in new_dp or new_dp[key] < new_cnt:
                    new_dp[key] = new_cnt
        dp = new_dp
    dp_max = max(dp.values()) if dp else 0  # maximum safe prefix count (m)
    
    # Finally, if m (dp_max) is less than N then we know that there is at least one dish that 
    # can be used as the extra (final) dish – it doesn’t matter if that extra dish causes a violation.
    # So the best achievable eaten-dish count is:
    #     ans = m + 1    if m < N, else m (if all N dishes can be arranged safely).
    if dp_max < N:
        ans = dp_max + 1
    else:
        ans = dp_max
        
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()