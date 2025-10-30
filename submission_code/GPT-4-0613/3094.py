from typing import List
from collections import Counter
import heapq

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counter = Counter(nums)
        heap = [-val for val in counter.values()]
        heapq.heapify(heap)
        operations = 0
        while len(heap) > 1:
            top = -heapq.heappop(heap)
            if top == 1:
                return -1
            elif top == 2:
                operations += 1
            else:
                heapq.heappush(heap, -(top - 2))
                operations += 1
        if heap:
            top = -heapq.heappop(heap)
            operations += top // 3
            if top % 3 != 0:
                return -1
        return operations