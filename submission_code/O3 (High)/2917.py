from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        """
        Counts the number of index pairs (i, j) such that:
          * 0 <= i < j < len(nums)
          * nums[i] + nums[j] < target

        With n capped at 50, a simple O(n²) double loop is sufficient.
        """
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] < target:
                    count += 1
        return count