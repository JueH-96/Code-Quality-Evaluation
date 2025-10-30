from typing import List
from collections import defaultdict

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        index = defaultdict(list)
        for i in range(2 * n):
            index[nums[i % n]].append(i)
        intervals = [(v[i], v[(i + 1) % len(v)]) for v in index.values() for i in range(len(v))]
        intervals.sort()
        max_interval = max(interval[1] - interval[0] for interval in intervals)
        max_interval = max(max_interval, intervals[0][0] + n - intervals[-1][1])
        return (n - max_interval) // 2