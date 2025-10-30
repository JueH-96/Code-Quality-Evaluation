from typing import List
import heapq

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = sorted(nums)
        result = []
        min_heap = []

        j = 0
        for i in range(n):
            # Maintain the min-heap with elements that can be swapped with nums[i]
            while j < n and (not min_heap or sorted_nums[j] - min_heap[0] <= limit):
                heapq.heappush(min_heap, sorted_nums[j])
                j += 1

            # Find the smallest element in the heap that can be swapped with nums[i]
            while min_heap and (min_heap[0] - nums[i] > limit or min_heap[0] < nums[i]):
                heapq.heappop(min_heap)

            if min_heap:
                result.append(heapq.heappop(min_heap))
            else:
                result.append(nums[i])

        return result