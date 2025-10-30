from typing import List
import heapq

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # Min-heap to store (nums[i], i)
        min_heap = []
        # Max-heap to store (-nums[i], i)
        max_heap = []
        min_diff = float('inf')

        for i in range(x):
            heapq.heappush(min_heap, (nums[i], i))
            heapq.heappush(max_heap, (-nums[i], i))

        for i in range(x, len(nums)):
            # Push the current element into both heaps
            heapq.heappush(min_heap, (nums[i], i))
            heapq.heappush(max_heap, (-nums[i], i))

            # Remove elements that are out of the window of size x
            while min_heap[0][1] <= i - x:
                heapq.heappop(min_heap)
            while -max_heap[0][0] < min_heap[0][0]:
                heapq.heappop(max_heap)
            while max_heap[0][1] <= i - x:
                heapq.heappop(max_heap)

            # Calculate the minimum difference
            min_diff = min(min_diff, -max_heap[0][0] - min_heap[0][0])

        return min_diff