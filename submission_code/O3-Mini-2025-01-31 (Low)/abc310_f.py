def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    mod = 998244353
    n = int(data[0])
    A = list(map(int, data[1:]))

    # Total number of outcomes is a product of A_i.
    total = 1
    for a in A:
        total = (total * a) % mod

    # We want to count, among outcomes, those for which one can choose a subset (of dice results)
    # whose sum is exactly 10.
    # Equivalently, we can count the "bad" outcomes that DO NOT allow any subset summing to 10 
    # and then subtract from total.
    #
    # A key observation: For any die with maximum A_i > 10, if its outcome is > 10 then clearly
    # it can never be used (since all picks are positive) toward making the sum 10 – we are free not 
    # to use any die. Only outcomes in the range 1..10 can possibly be used.
    #
    # For each die i we define:
    #   m = min(A_i, 10)          -- number of outcomes that are at most 10 (“contributory outcomes”)
    #   X = A_i - m               -- outcomes that are > 10, which always remain "unused" in the subset sum.
    #
    # Now, for each die, regardless of its outcome (even if ≤10) when deciding on a subset,
    # we are allowed to "not use" that die. But if its outcome is ≤10, there is also the choice to 
    # “use” the outcome.
    #
    # We now count the number of outcome combinations (over all dice) that do NOT allow any subset which 
    # sums to exactly 10. To do so, we build a DP that “simulates” the process of adding dice contributions
    # (if chosen) but we deliberately drop any branch that ever reaches a sum of exactly 10.
    #
    # We keep two kinds of states:
    #   dp_less[s] for s in [0, 9] – branches where the best achievable subset sum (by some choice of dice among those processed)
    #                             is exactly s (and we have not reached 10 yet).
    #   dp_more — a “bucket” for branches that have overshot 10 (i.e. some chosen dice produced a sum greater than 10);
    #             once overshot 10, we can never form exactly 10 (since all contributions are positive).
    #
    # We process dice one by one. For each die, we consider its two categories:
    #  1. Non-contributory outcomes: count = X. They do not allow the possibility of “using” any value – the only option
    #     is to “not use” that die, and the state remains the same.
    #  2. Contributory outcomes (values 1..m): For these, the outcome can be “ignored” (as in not choosing that die) 
    #     OR “used” i.e. if chosen in the subset, add its value.
    #
    # When updating a DP state from a previous state s (or state "more"), the transitions are as follows:
    #
    # For a die with parameters X and m:
    #
    # • For a state s (with s < 10):
    #   - Not using the die (or if its outcome is non-contributory) gives a multiplier (X + m)
    #     because regardless of whether the outcome is > 10 or ≤10, if you do not “use” the die, 
    #     the state remains unchanged.
    #   - Using the die: for each contributory outcome v from 1 to m,
    #       if s + v == 10, this branch reaches exactly 10 and is considered a "good" outcome – so we
    #       do not count it in our "bad" DP.
    #       if s + v < 10, then the new state is dp_less[s+v] increased by dp[s].
    #       if s + v > 10, then the branch goes to dp_more.
    #
    # • For a state "more" (already > 10):
    #   - Not using the die again: multiplies by (X + m) (and state remains in "more").
    #   - Using the die: any contributory outcome will keep the state > 10. There are m ways for that.
    #
    # At the end, the total count of "bad" outcomes is the sum of all states (dp_less for s in 0..9 and dp_more).
    # Then the count of "good" outcomes is:
    #        good = total - bad.
    # Finally, the probability answer is good/total modulo mod.
    #
    # We now implement this DP:
    dp_less = [0] * 10  # states for sums 0,1,...,9
    dp_less[0] = 1
    dp_more = 0

    for a in A:
        X = a - min(a, 10)  # outcomes > 10
        m = min(a, 10)      # outcomes in 1..10
        new_dp_less = [0] * 10
        new_dp_more = 0

        # Process transitions for states with sum < 10
        for s in range(10):
            ways = dp_less[s]
            if ways == 0:
                continue
            # Option1: Not using the die.
            #  This covers both: outcome >10 (X ways) or outcome ≤10 but not chosen (m ways).
            new_dp_less[s] = (new_dp_less[s] + ways * (X + m)) % mod

            # Option2: Using the die (i.e. choose the contributory outcome).
            # For each contributory outcome v, add if it does NOT produce exactly 10.
            for v in range(1, m + 1):
                ns = s + v
                if ns == 10:
                    # This branch produces sum 10, a "good" outcome; do not add to "bad" count.
                    continue
                elif ns < 10:
                    new_dp_less[ns] = (new_dp_less[ns] + ways) % mod
                else:  # ns > 10
                    new_dp_more = (new_dp_more + ways) % mod

        # Process transitions for state "more"
        if dp_more:
            # For a branch already > 10, any further outcomes keep it > 10.
            # Not using the die: factor = (X + m)
            new_dp_more = (new_dp_more + dp_more * (X + m)) % mod
            # Using the die: there are m contributory outcomes, each preserving "more".
            new_dp_more = (new_dp_more + dp_more * m) % mod

        dp_less, dp_more = new_dp_less, new_dp_more

    # Count the "bad" outcomes: ones which yield no subset with sum 10.
    dp_bad = (sum(dp_less) + dp_more) % mod
    good = (total - dp_bad) % mod

    # Since the probability is good/total (and total is invertible modulo mod),
    # we output good * inv(total) mod mod.
    inv_total = pow(total, mod - 2, mod)
    result = (good * inv_total) % mod
    sys.stdout.write(str(result))
    
if __name__ == "__main__":
    main()