from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        """
        Counts the number of pairs (i, j) such that 0 <= i < j < n
        and nums[i] + nums[j] < target.

        Args:
            nums: A 0-indexed integer array.
            target: An integer target value.

        Returns:
            The number of pairs (i, j) satisfying the conditions.
        """
        n = len(nums)
        count = 0
        
        # Iterate through all possible pairs (i, j) with i < j
        for i in range(n):
            # The second index j starts from i + 1 to ensure i < j
            for j in range(i + 1, n):
                # Check if the sum of elements at indices i and j is less than the target
                if nums[i] + nums[j] < target:
                    # If the condition is met, increment the counter
                    count += 1
                    
        return count