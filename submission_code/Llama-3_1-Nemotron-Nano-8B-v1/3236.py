from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 1  # As per constraints, nums is non-empty, but handle empty case gracefully
        
        # Determine the length of the longest sequential prefix
        k = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                k += 1
            else:
                break
        
        # Calculate the sum of the longest sequential prefix
        sum_seq = (nums[0] + (nums[0] + k - 1)) * k // 2
        
        # Create a set of the numbers for O(1) lookups
        num_set = set(nums)
        
        # Find the smallest x >= sum_seq not in the set
        x = sum_seq
        while x in num_set:
            x += 1
        
        return x