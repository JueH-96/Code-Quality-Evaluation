class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        L = len(s)
        # Precompute f[i] = number of steps to reduce i to 1 via popcount-iteration
        # f(1) = 0; for i >= 2: f(i) = 1 + f(popcount(i))
        max_i = L  # we only need up to L ones
        f = [0] * (max_i + 1)
        # f[0] is unused (we won't count x=0)
        f[1] = 0
        for i in range(2, max_i + 1):
            pc = bin(i).count('1')
            f[i] = 1 + f[pc]
        # Determine which counts of 1-bits are acceptable:
        # For x>=2, first step takes 1, so need f(popcount(x)) <= k-1.
        # For x=1 (popcount=1), f(1)=0, so included if 0 <= k-1 => k>=1 (always true here).
        acceptable = [False] * (max_i + 1)
        for m in range(1, max_i + 1):
            if f[m] <= k - 1:
                acceptable[m] = True

        # Digit DP over bits of s to count numbers < s with a given popcount.
        # dp[ones][tight]: number of ways for current prefix
        dp = [[0, 0] for _ in range(L + 1)]
        # start with zero bits chosen, tight=1 (we match s so far)
        dp[0][1] = 1
        for pos in range(L):
            bit_val = int(s[pos])
            new_dp = [[0, 0] for _ in range(L + 1)]
            for ones in range(0, pos + 1):
                for tight in (0, 1):
                    ways = dp[ones][tight]
                    if not ways:
                        continue
                    # we can put 0
                    nt = tight and (0 == bit_val)
                    new_dp[ones][0 if (tight and bit_val == 1 and 0 < bit_val) else nt] = (
                        new_dp[ones][0 if (tight and bit_val == 1 and 0 < bit_val) else nt] + ways
                    ) % mod
                    # we can put 1 if allowed
                    if tight:
                        if bit_val == 0:
                            # can't put 1, would exceed
                            continue
                        # bit_val == 1, putting 1 keeps tight
                        new_dp[ones + 1][1] = (new_dp[ones + 1][1] + ways) % mod
                    else:
                        # not tight, can put 1 freely
                        new_dp[ones + 1][0] = (new_dp[ones + 1][0] + ways) % mod
            dp = new_dp

        # Sum up all numbers < n (tight=0) with acceptable popcount >=1
        ans = 0
        for m in range(1, L + 1):
            if acceptable[m]:
                ans = (ans + dp[m][0]) % mod
        return ans