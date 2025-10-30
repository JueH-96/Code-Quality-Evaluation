from typing import List
import heapq

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        heap = nums.copy()
        heapq.heapify(heap)
        arr = []
        while heap:
            alice = heapq.heappop(heap)
            bob = heapq.heappop(heap)
            arr.append(bob)
            arr.append(alice)
        return arr