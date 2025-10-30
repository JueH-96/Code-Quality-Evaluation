from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        bit_count = [0] * 32
        
        # Count the number of 1s in each bit position
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Construct the k largest numbers
        result = 0
        for _ in range(k):
            num = 0
            for i in range(32):
                if bit_count[i] > 0:
                    num |= (1 << i)
                    bit_count[i] -= 1
            result = (result + (num * num) % MOD) % MOD
        
        return result