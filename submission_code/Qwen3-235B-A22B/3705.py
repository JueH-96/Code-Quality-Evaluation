from typing import List
from collections import defaultdict

class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        count = defaultdict(int)
        n = len(nums)
        # Iterate through all possible subarrays of size k
        for i in range(n - k + 1):
            subarray = nums[i:i+k]
            unique_elements = set(subarray)
            for num in unique_elements:
                count[num] += 1
        # Collect all elements that are 'almost missing'
        candidates = [x for x in count if count[x] == 1]
        return max(candidates) if candidates else -1