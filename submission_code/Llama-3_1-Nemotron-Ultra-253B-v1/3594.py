from typing import List
from collections import Counter

class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        count = Counter(nums)
        possible = set()
        for num in nums:
            candidate = total_sum - 2 * num
            if candidate in count:
                if num == candidate:
                    if count[num] >= 2:
                        possible.add(candidate)
                else:
                    possible.add(candidate)
        return max(possible)