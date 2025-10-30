from typing import List
from collections import defaultdict

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        dp = defaultdict(int)
        dp[(0, 0, 0, 0)] = 0  # Initial state: 0 elements in both groups, ORs are 0
        
        for num in nums:
            new_dp = defaultdict(int)
            for (a, b, or1, or2), current_xor in dp.items():
                # Option 1: Skip the current number
                new_dp[(a, b, or1, or2)] = max(new_dp[(a, b, or1, or2)], current_xor)
                
                # Option 2: Add to the first group
                if a < k:
                    new_a = a + 1
                    new_or1 = or1 | num
                    new_state = (new_a, b, new_or1, or2)
                    new_xor = new_or1 ^ or2
                    new_dp[new_state] = max(new_dp[new_state], new_xor)
                
                # Option 3: Add to the second group
                if b < k:
                    new_b = b + 1
                    new_or2 = or2 | num
                    new_state = (a, new_b, or1, new_or2)
                    new_xor = or1 ^ new_or2
                    new_dp[new_state] = max(new_dp[new_state], new_xor)
            dp = new_dp
        
        max_xor = 0
        for (a, b, or1, or2), current_xor in dp.items():
            if a == k and b == k:
                max_xor = max(max_xor, current_xor)
        return max_xor