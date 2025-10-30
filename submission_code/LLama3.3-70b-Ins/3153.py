from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort(reverse=True)
        res = sum([num**2 for num in nums[:k]])
        return res % MOD