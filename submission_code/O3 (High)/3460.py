from typing import List

MOD = 1_000_000_007


class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # convert “end” to prefix length (end is 0-based, length = end+1)
        req = {end + 1: cnt for end, cnt in requirements}

        # there must be a requirement for the whole permutation
        if n not in req:
            return 0

        # simple consistency checks that can give an early 0
        prev = -1
        for length in sorted(req):
            cnt = req[length]
            # impossible if more inversions than the prefix can contain
            if cnt > length * (length - 1) // 2:
                return 0
            # inversion count of longer prefix cannot be smaller
            if cnt < prev:
                return 0
            prev = cnt

        max_inv = req[n]                      # largest inversion value we ever have to handle
        dp = [0] * (max_inv + 1)              # dp[inv] – number of ways for current prefix length
        dp[0] = 1                             # empty prefix

        # iterate over positions we are going to fill
        for k in range(n):                    # k = current prefix length (before adding new element)
            new_dp = [0] * (max_inv + 1)
            window = 0                        # sliding window sum

            # after adding element #k, window size = k+1 (a_k can be 0 … k)
            for inv in range(max_inv + 1):
                window += dp[inv]
                if inv >= k + 1:
                    window -= dp[inv - (k + 1)]
                window %= MOD
                new_dp[inv] = window

            length = k + 1                    # new prefix length

            # apply requirement for this length, if it exists
            if length in req:
                required = req[length]
                if required > max_inv:        # should never happen, but be safe
                    return 0
                keep = new_dp[required] % MOD
                new_dp = [0] * (max_inv + 1)
                new_dp[required] = keep

            dp = new_dp

        # at length n we must have exactly req[n] inversions
        return dp[req[n]] % MOD