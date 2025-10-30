import bisect
from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        prev_sum = 0
        start = 0
        count = 0
        n = len(nums)
        while start < n:
            target = prefix[start] + prev_sum
            index = bisect.bisect_left(prefix, target, start + 1, len(prefix))
            if index > len(prefix) - 1:
                break
            count += 1
            prev_sum = prefix[index] - prefix[start]
            start = index
        return count