from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Count how many times each bit appears across all numbers
        MAXB = 31  # nums[i] <= 1e9 fits in 31 bits (0..30)
        bit_count = [0] * MAXB
        for x in nums:
            for b in range(MAXB):
                if (x >> b) & 1:
                    bit_count[b] += 1

        # We will build the k largest final values f[0] >= f[1] >= ... >= f[k-1].
        # By the analysis, the i-th largest final value is
        #   f[i] = sum_{p: bit_count[p] >= i+1} (1 << p).
        # Equivalently we can build an array f of length k by
        # adding (1<<p) to f[0], f[1], ..., f[bit_count[p]-1] (capped at k).
        f = [0] * k
        for p in range(MAXB):
            cnt = bit_count[p]
            # We can only assign this bit to at most cnt of the k numbers,
            # and at most k in total, so use = min(cnt, k).
            use = cnt if cnt < k else k
            if use > 0:
                val = 1 << p
                for i in range(use):
                    f[i] += val

        # Now f is sorted descending by construction: f[0] >= f[1] >= ...
        # The answer is sum of squares of these k values, mod 1e9+7.
        ans = 0
        for x in f:
            xm = x % MOD
            ans = (ans + xm * xm) % MOD
        return ans