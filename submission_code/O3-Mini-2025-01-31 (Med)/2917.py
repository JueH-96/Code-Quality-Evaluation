from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        count = 0
        n = len(nums)
        # Iterate over all unique pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] < target:
                    count += 1
        return count