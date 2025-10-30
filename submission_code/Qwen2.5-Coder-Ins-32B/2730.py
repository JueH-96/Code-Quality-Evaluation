from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * n
        suffix_or = [0] * n
        
        # Calculate prefix OR values
        for i in range(1, n):
            prefix_or[i] = prefix_or[i - 1] | nums[i - 1]
        
        # Calculate suffix OR values
        for i in range(n - 2, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | nums[i + 1]
        
        max_value = 0
        for i in range(n):
            # Calculate the new value if we multiply nums[i] by 2^k
            new_value = nums[i] << k
            # Calculate the new OR value for the array
            current_or = prefix_or[i] | new_value | suffix_or[i]
            # Update the maximum OR value
            max_value = max(max_value, current_or)
        
        return max_value