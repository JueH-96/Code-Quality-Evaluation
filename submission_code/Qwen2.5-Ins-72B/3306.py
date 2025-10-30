from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        min_heap = [(val, idx) for idx, val in enumerate(nums)]
        heapq.heapify(min_heap)
        total_sum = sum(nums)
        result = []

        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                total_sum -= nums[index]

            while k > 0 and min_heap:
                val, idx = heapq.heappop(min_heap)
                if not marked[idx]:
                    marked[idx] = True
                    total_sum -= nums[idx]
                    k -= 1

            result.append(total_sum)

        return result