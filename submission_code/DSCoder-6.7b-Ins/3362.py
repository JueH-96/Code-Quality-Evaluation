from typing import List
from collections import defaultdict
from bisect import bisect_left, bisect_right

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        num_freq = defaultdict(int)
        unique_counts = []

        for i in range(n):
            num_freq[nums[i]] += 1
            unique_counts.append(len(num_freq))

        unique_counts.sort()
        median_idx = n // 2

        if n % 2 == 0:
            return (unique_counts[median_idx - 1] + unique_counts[median_idx]) / 2
        else:
            return unique_counts[median_idx]

# Test cases
sol = Solution()
print(sol.medianOfUniquenessArray([1,2,3])) # 1
print(sol.medianOfUniquenessArray([3,4,3,4,5])) # 2
print(sol.medianOfUniquenessArray([4,3,5,4])) # 2