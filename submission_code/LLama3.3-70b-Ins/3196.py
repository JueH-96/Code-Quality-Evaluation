from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        max_score = 0
        for i in range(len(nums)):
            total = 0
            for j in range(i, len(nums)):
                total += nums[j] - nums[i]
                if total <= k:
                    max_score = max(max_score, j - i + 1)
                else:
                    break
        return max_score