def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Parse input.
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
        K = int(next(it))
    except StopIteration:
        return
    mod = 998244353

    # In the graph, besides the “cycle‐edge” which always moves from vertex v to v+1 (cyclically)
    # we have M extra edges. For an extra edge from vertex X to vertex Y, notice that if one were to take
    # the cycle edge at vertex X the destination would have been X+1.
    # Thus when taking the extra edge instead one "jumps" to Y.
    # We encode our state as an “offset” r (mod N) such that after t moves the current vertex is
    #   vertex = ((t + r) mod N) + 1.
    # Without any extra move, we always would have r = 0.
    # Now, say at some move t we are at a state with offset r.
    # The extra edge is available if
    #      ((t + r) mod N) + 1 == X,
    # i.e. if (t + r) mod N == X-1.
    # If one takes that extra edge instead of the cycle edge, then one “misses” the normal +1 move
    # and instead is “teleported” to Y.
    # Hence the offset changes by
    #      d = (Y - (X+1)) mod N.
    #
    # We now pre‐compute for each extra edge its parameters:
    #   X_minus = X-1  (so that condition is: (t + r) mod N == X_minus)
    # and d = (Y - (X+1)) mod N.
    #
    # On any move t (starting from t=0) if our state dp (an array indexed by r, modulo N) has
    # a contribution at index r with (t + r) mod N == X_minus then one may “jump” and add that count
    # to state dp at index (r+d) mod N.
    #
    # We note that the cycle move “copies” the state over (since if no extra edge is taken
    # then the offset remains unchanged), so the recurrence is:
    # 
    #   f(0)[r] = 1 if r == 0, else 0.
    #   For t = 0 .. K-1, let f(t+1) = f(t) + (contribution from extra edges applied to f(t)).
    #
    # And the contribution for each extra edge (X,Y) is:
    #    if (t + r) mod N == (X-1) then add f(t)[r] to f(t+1)[ (r + d) mod N ].
    #
    # We must compute answer = sum_r f(K)[r] mod mod.
    #
    # Notice that even though t appears in the condition we can rephrase it:
    # given a current move t, let t_mod = (t mod N). Then the condition becomes:
    #    r == (X-1 - t_mod) mod N.
    # For every extra edge we precompute its starting value X_minus and its offset d.
    
    edges = []
    for _ in range(M):
        try:
            X = int(next(it))
            Y = int(next(it))
        except StopIteration:
            break
        # For an extra edge from X to Y, condition is that current vertex equals X.
        # Since current vertex = ((t + r) mod N) + 1, we need (t + r) mod N == X-1.
        # We store X_minus = X - 1.
        # And if this extra edge is taken, the offset is increased by:
        #   d = (Y - (X+1)) mod N.
        X_minus = X - 1
        d = (Y - (X + 1)) % N
        edges.append((X_minus, d))
    
    # dp[r] will hold f(K)[r] (the cumulative count for state (offset) r)
    # Initially, f(0)[0] = 1.
    dp = [0] * N
    dp[0] = 1

    # We perform K moves (t = 0 to K-1).
    # The cycle move “copies” the state along (the state never subtracts) so when taking extra edges
    # we simply add contributions based on the current cumulative state.
    # At move t, for each extra edge (X_minus, d), if
    #     r == (X_minus - t_mod) mod N,
    # then f(t)[r] contributes to f(t+1)[(r+d) mod N].
    #
    # We use t_mod to represent (t mod N) and update it each step.
    t_mod = 0
    N_val = N
    mod_val = mod
    dp_list = dp  # local alias for speed
    for _ in range(K):
        # For each extra edge try to add the jump contribution.
        for (X_minus, d) in edges:
            # Compute the required index in dp: we need r = (X_minus - t_mod) mod N.
            r = X_minus - t_mod
            if r < 0:
                r += N_val
            # Get the count at that index.
            cnt = dp_list[r]
            if cnt:
                # The jump adds this same count to dp at index (r + d) mod N.
                pos = r + d
                if pos >= N_val:
                    pos -= N_val
                newval = dp_list[pos] + cnt
                if newval >= mod_val:
                    newval -= mod_val
                dp_list[pos] = newval
        # Advance t_mod = (t+1) mod N.
        t_mod += 1
        if t_mod >= N_val:
            t_mod -= N_val

    # The answer is the sum of all dp entries mod mod.
    ans = 0
    for num in dp_list:
        ans = (ans + num) % mod_val
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()