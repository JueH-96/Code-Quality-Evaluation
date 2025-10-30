from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # dp will map a bitwise-AND value to the count of subarrays ending at the previous index
        dp = {}   # type: Dict[int, int]
        total = 0
        
        for num in nums:
            new_dp = defaultdict(int)
            # Start a new subarray at this position
            new_dp[num] += 1
            
            # Extend all previous subarrays by this element
            for val, cnt in dp.items():
                new_val = val & num
                new_dp[new_val] += cnt
            
            # Count how many end with AND == k
            total += new_dp.get(k, 0)
            dp = new_dp
        
        return total