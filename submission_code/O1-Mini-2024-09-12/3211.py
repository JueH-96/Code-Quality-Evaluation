from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        suffix_sums = []
        for num in reversed(nums):
            current_sum = num
            while suffix_sums and current_sum > suffix_sums[0]:
                current_sum += suffix_sums.pop(0)
            suffix_sums.insert(0, current_sum)
        return len(suffix_sums)