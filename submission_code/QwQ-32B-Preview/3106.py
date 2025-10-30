from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        previous = [-1] * (target + 1)
        previous[0] = 0  # Base case: empty subsequence sums to 0
        
        for num in nums:
            current = previous.copy()
            for j in range(target - num + 1):
                if previous[j] != -1:
                    current[j + num] = max(current[j + num], previous[j] + 1)
            previous = current
        
        return previous[target] if previous[target] != -1 else -1