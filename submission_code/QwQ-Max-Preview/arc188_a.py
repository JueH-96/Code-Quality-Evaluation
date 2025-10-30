MOD = 998244353

def main():
    import sys
    from collections import defaultdict

    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # Initialize DP: key is (a, b, c), value is a dictionary of {sum: count}
    dp = defaultdict(lambda: defaultdict(int))
    dp[(0, 0, 0)][0] = 1

    for j in range(N):
        char = S[j]
        new_dp = defaultdict(lambda: defaultdict(int))

        # Compute prev_total: sum of all counts for each (a_prev, b_prev, c_prev)
        prev_total = defaultdict(int)
        for (a_prev, b_prev, c_prev) in dp:
            total = sum(dp[(a_prev, b_prev, c_prev)].values()) % MOD
            prev_total[(a_prev, b_prev, c_prev)] = (prev_total[(a_prev, b_prev, c_prev)] + total) % MOD

        # Precompute added_sums for all possible (a, b, c)
        added_sums = defaultdict(int)
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    d0_key = (a, b, c)
                    d1_a = (a - 1) % 2
                    d1_b = (b - 1) % 2
                    d1_c = (c - 1) % 2
                    d1_key = (d1_a, d1_b, d1_c)
                    added = (prev_total.get(d0_key, 0) + prev_total.get(d1_key, 0)) % MOD
                    added_sums[(a, b, c)] = added

        # Process each state in the current DP
        for (a_prev, b_prev, c_prev) in list(dp.keys()):
            for s_prev in list(dp[(a_prev, b_prev, c_prev)].keys()):
                count_prev = dp[(a_prev, b_prev, c_prev)][s_prev]

                if char == 'A':
                    a_new = (a_prev + 1) % 2
                    added = added_sums.get((a_new, b_prev, c_prev), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_new, b_prev, c_prev)][s_new] = (new_dp[(a_new, b_prev, c_prev)][s_new] + count_prev) % MOD
                elif char == 'B':
                    b_new = (b_prev + 1) % 2
                    added = added_sums.get((a_prev, b_new, c_prev), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_prev, b_new, c_prev)][s_new] = (new_dp[(a_prev, b_new, c_prev)][s_new] + count_prev) % MOD
                elif char == 'C':
                    c_new = (c_prev + 1) % 2
                    added = added_sums.get((a_prev, b_prev, c_new), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_prev, b_prev, c_new)][s_new] = (new_dp[(a_prev, b_prev, c_new)][s_new] + count_prev) % MOD
                else:  # '?'
                    # Case 1: replace with 'A'
                    a_new = (a_prev + 1) % 2
                    added = added_sums.get((a_new, b_prev, c_prev), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_new, b_prev, c_prev)][s_new] = (new_dp[(a_new, b_prev, c_prev)][s_new] + count_prev) % MOD
                    # Case 2: replace with 'B'
                    b_new = (b_prev + 1) % 2
                    added = added_sums.get((a_prev, b_new, c_prev), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_prev, b_new, c_prev)][s_new] = (new_dp[(a_prev, b_new, c_prev)][s_new] + count_prev) % MOD
                    # Case 3: replace with 'C'
                    c_new = (c_prev + 1) % 2
                    added = added_sums.get((a_prev, b_prev, c_new), 0)
                    s_new = (s_prev + added) % MOD
                    new_dp[(a_prev, b_prev, c_new)][s_new] = (new_dp[(a_prev, b_prev, c_new)][s_new] + count_prev) % MOD

        dp = new_dp

    # Calculate the answer
    answer = 0
    for (a, b, c) in dp:
        for s in dp[(a, b, c)]:
            if s >= K:
                answer = (answer + dp[(a, b, c)][s]) % MOD

    print(answer % MOD)

if __name__ == "__main__":
    main()