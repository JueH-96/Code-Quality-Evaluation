from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        previous = [0] + [-1] * target
        for num in nums:
            current = previous.copy()
            for j in range(num, target + 1):
                if previous[j - num] != -1:
                    current[j] = max(current[j], previous[j - num] + 1)
            previous = current
        return previous[target] if previous[target] != -1 else -1