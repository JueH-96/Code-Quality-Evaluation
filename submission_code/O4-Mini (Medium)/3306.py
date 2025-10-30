import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # marked flags
        marked = [False] * n
        # Min-heap of (value, index) for all elements
        heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(heap)
        # Current sum of unmarked elements
        total_unmarked_sum = sum(nums)
        answers = []
        
        for idx, k in queries:
            # 1) Mark the element at index idx if not already marked
            if not marked[idx]:
                marked[idx] = True
                total_unmarked_sum -= nums[idx]
            # 2) Mark k smallest unmarked elements
            for _ in range(k):
                # Pop until we find an unmarked one or heap is empty
                while heap and marked[heap[0][1]]:
                    heapq.heappop(heap)
                if not heap:
                    break
                val, j = heapq.heappop(heap)
                # Mark it
                marked[j] = True
                total_unmarked_sum -= val
            # Record current sum of unmarked elements
            answers.append(total_unmarked_sum)
        
        return answers