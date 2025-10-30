import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        marked = [False] * len(nums)
        heap = [(num, idx) for idx, num in enumerate(nums)]
        heapq.heapify(heap)
        result = []
        total_sum = sum(nums)
        for index, k in queries:
            if not marked[index]:
                total_sum -= nums[index]
                marked[index] = True
            for _ in range(k):
                if heap:
                    num, idx = heapq.heappop(heap)
                    if not marked[idx]:
                        total_sum -= num
                        marked[idx] = True
            result.append(total_sum)
        return result