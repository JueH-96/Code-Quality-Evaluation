from typing import List
import heapq

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        heap = []
        total = 0
        groups = 0
        for limit in usageLimits:
            total += limit
            heapq.heappush(heap, -limit)
            while total >= 0:
                total += heapq.heappop(heap)
                groups += 1
        return groups