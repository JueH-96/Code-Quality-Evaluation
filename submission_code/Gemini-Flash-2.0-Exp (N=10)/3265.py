from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = 0
        found_good = False
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if abs(nums[i] - nums[j]) == k:
                    current_sum = sum(nums[i:j+1])
                    if not found_good:
                        max_sum = current_sum
                        found_good = True
                    else:
                        max_sum = max(max_sum, current_sum)
        return max_sum