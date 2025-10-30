from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        bit_counts = [0] * 31
        
        for num in nums:
            for b in range(31):
                if num & (1 << b):
                    bit_counts[b] += 1
        
        res = [0] * n
        for b in range(30, -1, -1):
            cnt = bit_counts[b]
            for i in range(cnt):
                res[i] += 1 << b
        
        total = 0
        for i in range(k):
            total = (total + res[i] * res[i]) % MOD
        
        return total