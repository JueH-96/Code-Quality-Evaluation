from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_value = 0
        
        # Iterate over all possible subsequences of size 2*k
        for i in range(n - 2 * k + 1):
            # Calculate OR for the first half
            or1 = 0
            for j in range(i, i + k):
                or1 |= nums[j]
            
            # Calculate OR for the second half
            or2 = 0
            for j in range(i + k, i + 2 * k):
                or2 |= nums[j]
            
            # Calculate XOR of the two OR results
            current_value = or1 ^ or2
            
            # Update the maximum value found
            max_value = max(max_value, current_value)
        
        return max_value