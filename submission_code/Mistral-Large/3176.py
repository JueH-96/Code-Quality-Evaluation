from typing import List

class Solution:
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        min_sum = float('inf')
        found = False

        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                if nums[i] < nums[j]:
                    for k in range(j + 1, n):
                        if nums[k] < nums[j]:
                            found = True
                            min_sum = min(min_sum, nums[i] + nums[j] + nums[k])

        return min_sum if found else -1