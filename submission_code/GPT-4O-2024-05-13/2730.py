from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Calculate the initial OR of all elements
        initial_or = 0
        for num in nums:
            initial_or |= num
        
        max_or = initial_or
        
        # Try multiplying each element by 2^k and calculate the OR
        for i in range(n):
            # Calculate the OR without the current element
            current_or = initial_or & ~nums[i]
            # Multiply the current element by 2^k
            modified_num = nums[i] << k
            # Calculate the new OR
            new_or = current_or | modified_num
            # Update the maximum OR
            max_or = max(max_or, new_or)
        
        return max_or