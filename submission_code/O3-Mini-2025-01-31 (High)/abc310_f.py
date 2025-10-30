def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    mod = 998244353

    # Read input values.
    N = int(data[0])
    A_list = list(map(int, data[1:]))

    # Total number of outcomes = A1 * A2 * ... * AN (mod mod)
    total = 1
    for a in A_list:
        total = (total * a) % mod

    # We want to count the configurations (i.e. the outcomes) for which
    # even when considering all dice that rolled a number in [1, min(a,10)]
    # (the "usable" outcomes), no subset sum equals exactly 10.
    # We use DP with states represented as bitmasks of length 11 (positions 0,...,10).
    # In a state bitmask, if bit j is set then there is some subset (of usable rolled numbers)
    # that sums to j.
    #
    # Initially only 0 is achievable.
    mask = (1 << 11) - 1  # This is 2047, we will keep only bits 0..10.
    dp = [0] * (1 << 11)
    init = 1  # Only bit0 is set (i.e. sum 0 achievable)
    dp[init] = 1

    # Process each die i.
    for a in A_list:
        # Usable outcomes: values 1 .. U, where U = min(a,10)
        U = a if a < 10 else 10
        # Non-usable outcomes: if a > 10 then there are a - 10 outcomes > 10.
        N_val = a - U if a > 10 else 0

        new_dp = [0] * (1 << 11)
        # Iterate over every DP state (s: a bitmask) that we reached so far.
        for s in range(1 << 11):
            ways = dp[s]
            if ways == 0:
                continue
            # Option 1: the die rolls a non-usable outcome.
            # In that case the state remains the same.
            if N_val:
                new_dp[s] = (new_dp[s] + ways * N_val) % mod
            # Option 2: the die rolls a usable outcome v (from 1 to U).
            # In that case, the "available" number v is added
            # which means the new achievable set becomes:
            #   new_state = s OR ((s << v) & mask)
            # If this new_state ends up with bit10 set then a subset summing to 10 exists;
            # in that branch, the configuration is successful and we do not want to count it
            # in the failure count.
            for v in range(1, U+1):
                ns = s | ((s << v) & mask)
                if ns & (1 << 10):
                    # This branch would allow a subset summing to 10, so skip it.
                    continue
                new_dp[ns] = (new_dp[ns] + ways) % mod
        dp = new_dp

    # fail_count = number of outcomes where no subset of usable dice sums to 10.
    fail_count = sum(dp) % mod
    success_count = (total - fail_count) % mod
    inv_total = pow(total, mod - 2, mod)
    ans = (success_count * inv_total) % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()