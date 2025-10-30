import typing
from typing import List

class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 1000000007
        n = len(nums)
        
        # Compute the count of each bit set across all numbers
        bit_count = [0] * 30
        for num in nums:
            for b in range(30):
                if num & (1 << b):
                    bit_count[b] += 1
        
        # Initialize the sums for the top k elements
        top_sums = [0] * k
        
        # Process each bit from highest to lowest
        for b in range(29, -1, -1):
            num_to_set = min(bit_count[b], k)
            if num_to_set == 0:
                continue
            value_add = 1 << b  # 2**b
            
            # Sort the current sums in descending order
            top_sums.sort(reverse=True)
            
            # Add the value to the num_to_set largest elements
            for i in range(num_to_set):
                top_sums[i] += value_add
        
        # Compute the sum of squares modulo MOD
        ans = 0
        for s in top_sums:
            sq = (s * s) % MOD
            ans += sq
            ans %= MOD
        
        return ans