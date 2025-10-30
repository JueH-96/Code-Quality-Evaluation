from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        total_or = 0
        for num in nums:
            total_or |= num
        
        bits = [0] * 32
        for num in nums:
            for i in range(32):
                if (num >> i) & 1:
                    bits[i] += 1
        
        result = 0
        MOD = 10**9 + 7
        
        for _ in range(k):
            current_num = 0
            for i in range(31, -1, -1):
                if bits[i] > 0:
                    current_num |= (1 << i)
                    bits[i] -= 1
            result = (result + (current_num * current_num) % MOD) % MOD
        
        return result