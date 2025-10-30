from collections import defaultdict
import bisect
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        groups = defaultdict(list)
        for idx, num in enumerate(nums):
            groups[num].append(idx)
        
        max_len = 0
        for x in groups:
            pos = groups[x]
            a = [p - i for i, p in enumerate(pos)]
            m = len(pos)
            for j in range(m):
                target = a[j] - k
                i = bisect.bisect_left(a, target)
                current = j - i + 1
                if current > max_len:
                    max_len = current
        return max_len