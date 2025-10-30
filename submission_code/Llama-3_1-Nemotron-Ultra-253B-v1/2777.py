from typing import List

class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_counts = []
        left_set = set()
        for num in nums:
            left_set.add(num)
            prefix_counts.append(len(left_set))
        
        suffix_counts = [0] * n
        right_set = set()
        for i in reversed(range(n)):
            right_set.add(nums[i])
            suffix_counts[i] = len(right_set)
        
        diff = []
        for i in range(n):
            suffix = suffix_counts[i + 1] if i + 1 < n else 0
            diff.append(prefix_counts[i] - suffix)
        
        return diff