class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        # We need compositions of n into r positive parts each <= limit.
        # Precompute comp[n][r] for n up to max(zero, one)
        N = max(zero, one)
        # comp[n][r] = number of ways to write n as sum of r parts, each 1..limit
        comp = [[0] * (N+1) for _ in range(N+1)]
        comp[0][0] = 1
        # Build up for n from 1..N, r from 1..n
        for n in range(1, N+1):
            for r in range(1, n+1):
                total = 0
                # last part k from 1 to limit (and <= n)
                maxk = min(limit, n)
                # sum comp[n-k][r-1] for k=1..maxk
                # We can loop
                for k in range(1, maxk+1):
                    total += comp[n-k][r-1]
                comp[n][r] = total % MOD

        # Now sum over run counts
        # sum_e: equal runs (a zero-runs and a one-runs) for start-zero or start-one
        sum_e = 0
        # sum_z: start with zero and zero-run count = one-run count+1 => a runs of zeros, a-1 runs of ones
        sum_z = 0
        # sum_o: start with one and one-run count = zero-run count+1 => a runs of ones, a-1 runs of zeros
        sum_o = 0

        # a is number of zero-runs when iterating sum_e and sum_z; for sum_o roles swap
        # sum_e: a from 1..min(zero, one)
        max_runs = N
        for a in range(1, max_runs+1):
            # equal runs case
            if a <= zero and a <= one:
                sum_e = (sum_e + comp[zero][a] * comp[one][a]) % MOD
            # start with zero and one-run count = a-1
            if a <= zero and (a-1) >= 0 and (a-1) <= one:
                sum_z = (sum_z + comp[zero][a] * comp[one][a-1]) % MOD
            # start with one and zero-run count = a-1
            if a <= one and (a-1) >= 0 and (a-1) <= zero:
                sum_o = (sum_o + comp[one][a] * comp[zero][a-1]) % MOD

        # total: start-zero equal-runs + start-one equal-runs => 2*sum_e
        ans = (2 * sum_e + sum_z + sum_o) % MOD
        return ans