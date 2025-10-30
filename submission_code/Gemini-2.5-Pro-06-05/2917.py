from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        """
        Counts the number of pairs (i, j) in a list such that i < j and nums[i] + nums[j] < target.

        Args:
            nums: A 0-indexed list of integers.
            target: An integer target value.

        Returns:
            The number of pairs (i, j) satisfying the conditions.
        """
        
        n = len(nums)
        count = 0
        
        # Iterate through each unique pair of indices (i, j) where i < j.
        # The outer loop iterates through the first element of the pair.
        for i in range(n):
            # The inner loop iterates through the second element of the pair,
            # starting from the index after i to ensure that j > i.
            for j in range(i + 1, n):
                # If the sum of the elements at these indices is less than the target,
                # increment the count.
                if nums[i] + nums[j] < target:
                    count += 1
                    
        return count