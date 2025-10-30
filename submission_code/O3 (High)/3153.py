from typing import List

MOD = 1_000_000_007          # required modulus
MAX_BIT = 31                 # 0 … 30 are enough for nums[i] ≤ 10^9


class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        # 1. how many ones does each bit position have?
        bit_cnt = [0] * MAX_BIT
        for num in nums:
            for b in range(MAX_BIT):
                if num >> b & 1:
                    bit_cnt[b] += 1

        # 2. build the k numbers that give the largest possible
        #    sum of squares.
        best = [0] * k                     # always kept non-increasing
        for b in range(MAX_BIT - 1, -1, -1):
            take = min(bit_cnt[b], k)      # at most one ‘1’ per number
            add  = 1 << b
            for i in range(take):          # put this bit into the prefix
                best[i] += add

        # 3. compute answer
        ans = 0
        for val in best:
            ans = (ans + (val % MOD) * (val % MOD)) % MOD
        return ans