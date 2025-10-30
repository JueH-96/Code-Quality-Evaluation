from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        marked = [False] * n
        min_heap = [(nums[i], i) for i in range(n)]
        heapq.heapify(min_heap)

        result = []
        current_sum = sum(nums)

        for index, k in queries:
            if not marked[index]:
                marked[index] = True
                current_sum -= nums[index]

            count = 0
            while min_heap and count < k:
                value, idx = heapq.heappop(min_heap)
                if not marked[idx]:
                    marked[idx] = True
                    current_sum -= value
                    count += 1

            result.append(current_sum)

        return result