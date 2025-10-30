from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = float('-inf')
        
        # Iterate over all possible triplets of indices (i, j, k) such that i < j < k
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    # Calculate the value of the current triplet
                    triplet_value = (nums[i] - nums[j]) * nums[k]
                    
                    # Update max_value if the current triplet value is greater
                    max_value = max(max_value, triplet_value)
        
        # If all triplets have a negative value, return 0
        return max_value if max_value > 0 else 0