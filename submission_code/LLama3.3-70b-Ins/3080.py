from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        def calculate_score(subarray: List[int]) -> int:
            """Calculate the score of a subarray by performing bitwise AND operation."""
            score = subarray[0]
            for num in subarray[1:]:
                score &= num
            return score

        n = len(nums)
        max_subarrays = 1
        current_score = nums[0]
        
        # Iterate through the array
        for i in range(1, n):
            # If the current number is 0, it will make the score of any subarray 0
            # So, we can start a new subarray from here
            if nums[i] == 0:
                max_subarrays += 1
                current_score = 0
            else:
                # If the current number is not 0, update the current score
                current_score &= nums[i]
        
        # If the current score is not 0, it means we can't split the array further
        # So, return 1
        if current_score != 0:
            return 1
        
        return max_subarrays