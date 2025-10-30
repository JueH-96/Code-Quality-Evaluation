from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        n = len(nums)

        # Count occurrences of each number in all subarrays of size k
        for i in range(n - k + 1):
            seen = set()
            for j in range(k):
                seen.add(nums[i + j])
            for num in seen:
                count[num] += 1

        # Find the largest integer that appears in exactly one subarray
        largest_almost_missing = -1
        for num, c in count.items():
            if c == 1:
                largest_almost_missing = max(largest_almost_missing, num)

        return largest_almost_missing