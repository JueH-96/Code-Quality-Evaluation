from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        n = len(nums)
        count = 0

        # Iterate over all possible pairs (i, j)
        # The outer loop for index i goes from 0 to n-1
        for i in range(n):
            # The inner loop for index j goes from i+1 to n-1
            # This ensures that i < j and each pair is considered only once
            for j in range(i + 1, n):
                # Check if the sum of the elements at these indices is less than the target
                if nums[i] + nums[j] < target:
                    count += 1
        
        return count