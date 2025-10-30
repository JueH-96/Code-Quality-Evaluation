from typing import List
import heapq

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        nums.sort()
        heap = []
        for num in nums:
            if heap and heap[0] < num:
                heapq.heappop(heap)
            heapq.heappush(heap, num + 1)
        return len(heap)