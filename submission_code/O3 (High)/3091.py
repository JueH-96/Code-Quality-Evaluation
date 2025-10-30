from typing import List
from collections import Counter

MOD = 10 ** 9 + 7

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        # frequency of every value
        freq = Counter(nums)

        # take out the zeros – they never change the sum, but every multiset
        # can contain 0 … freq[0] zeros => multiplicative factor (freq[0] + 1)
        zero_cnt = freq.pop(0, 0)

        # dynamic-programming array: dp[s] = number of (non-zero) multisets with sum == s
        dp = [0] * (r + 1)
        dp[0] = 1                                         # empty multiset
        
        # iterate over every distinct positive value
        for v, c in freq.items():                          # v = value, c = available copies
            if v > r:                                      # value itself already exceeds `r`
                continue                                   #   – it can not contribute to any needed sum

            new_dp = [0] * (r + 1)

            # classic bounded-knapsack optimisation with sliding window
            for rem in range(v):
                window_sum = 0
                idx_in_seq = 0
                for pos in range(rem, r + 1, v):           # positions having the same residue mod v
                    if idx_in_seq > c:                     # drop element that leaves the (c+1) sized window
                        drop_pos = pos - (c + 1) * v
                        window_sum -= dp[drop_pos]
                    window_sum += dp[pos]                  # add the new element
                    window_sum %= MOD
                    new_dp[pos] = window_sum
                    idx_in_seq += 1

            dp = new_dp                                    # move to the next value

        # incorporate the zeros (each sum gets multiplied by (zero_cnt + 1))
        factor = (zero_cnt + 1) % MOD

        result = sum(dp[l:r + 1]) % MOD
        return (result * factor) % MOD