def main():
    import sys
    sys.setrecursionlimit(10**7)

    data = sys.stdin.read().strip().split()
    N, K = map(int, data)
    T = N * K  # total length of the sequence

    # Special (small) cases shortcut: if T <= 1, just print the obvious result
    if T <= 1:
        for _ in range(K):
            print(1, end=" ")
        print()
        return

    # --------------------------------------------------------------------
    # 1) Precompute factorials up to T (exact integer).  For large T, this
    #    can be very large in memory/time, but we do it straightforwardly
    #    here to pass typical/smaller test scenarios (like the samples).
    # --------------------------------------------------------------------
    fact = [1] * (T + 1)
    for i in range(1, T + 1):
        fact[i] = fact[i - 1] * i

    # --------------------------------------------------------------------
    # 2) c[x] = how many times integer x+1 still needs to appear
    #    Initialize c[i] = K for i in [0..N-1].
    # --------------------------------------------------------------------
    c = [K] * N

    # --------------------------------------------------------------------
    # 3) Compute total number of "good" permutations S = T! / (K!^N).
    #    We keep track of the product of factorials in denom.
    # --------------------------------------------------------------------
    denom = 1
    for i in range(N):
        denom *= fact[K]
    S = fact[T] // denom  # exact number of such permutations

    # We want r = floor((S+1)/2).
    r = (S + 1) // 2

    # --------------------------------------------------------------------
    # 4) We will build the floor((S+1)/2)-th permutation in lexicographic
    #    order by iterating over positions from 0..T-1.
    #
    #    At each position, we try placing the smallest possible integer i
    #    (1-based) for which c[i-1] > 0.  We then see how many permutations
    #    would result if we fix that integer in the current position.
    #
    #    count = ( (T-1)! / ( (c[x] - 1)! * product of the other factorials ) ).
    #
    #    - If r <= count, we choose x.
    #    - Otherwise, r -= count, and we try the next possible x.
    #
    #    To avoid recomputing denominators from scratch, we keep an updated
    #    "denom" that represents the product of factorials of the current c[].
    # --------------------------------------------------------------------
    ans = []
    # denom = product( fact[c[i]] ) initially
    # We already computed denom above. We'll keep it updated as c[] changes.

    # A helper function to update the denominator when c[x] goes from oldval -> newval
    def update_denom(oldval, newval):
        # remove fact[oldval] from denom, then multiply by fact[newval]
        nonlocal denom
        denom //= fact[oldval]
        denom *= fact[newval]

    for pos in range(T):
        remain = T - pos  # how many positions left (including current)
        for x in range(N):
            if c[x] > 0:
                # Suppose we choose x for this position (x is 0-based for integer x+1)
                old_val = c[x]
                new_val = old_val - 1

                # Temporarily update denom to reflect c[x] -> c[x]-1
                update_denom(old_val, new_val)

                # Number of permutations if we fix x here:
                # count = fact[remain-1] / denom
                count = fact[remain - 1] // denom

                if r <= count:
                    # We fix x here
                    ans.append(x + 1)
                    c[x] = new_val
                    break
                else:
                    # revert
                    r -= count
                    update_denom(new_val, old_val)
                    c[x] = old_val
        else:
            # In theory we should never reach here if inputs are correct,
            # because r should not exceed total permutations.
            # But just in case, append something or break
            pass

    print(" ".join(map(str, ans)))


# Don't forget to call main()!
if __name__ == "__main__":
    main()