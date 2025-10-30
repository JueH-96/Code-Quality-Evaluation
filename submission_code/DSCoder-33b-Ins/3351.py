from typing import List
import heapq

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort()
        total_sum = 0
        max_heap = []
        for i in range(len(happiness) - 1, -1, -1):
            total_sum += happiness[i]
            if i > 0:
                heapq.heappush(max_heap, -happiness[i] + happiness[i - 1])
            if len(max_heap) > k - 1:
                total_sum += heapq.heappop(max_heap)
        return total_sum