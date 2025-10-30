from typing import List
import heapq

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        pairs = [(nums1[i], nums2[i]) for i in range(n)]
        pairs.sort(key=lambda x: (-x[0], -x[1]))

        max_heap = []
        max_y = 0
        sums = {}

        for x, y in pairs:
            heapq.heappush(max_heap, (y, x + y))
            max_y = max(max_y, y)
            while max_heap and max_heap[0][0] < max_y:
                heapq.heappop(max_heap)
            if max_heap:
                sums[x] = max_heap[0][1]

        result = []
        for x_i, y_i in queries:
            max_sum = -1
            for x in sums:
                if x >= x_i and sums[x] >= x_i + y_i:
                    max_sum = max(max_sum, sums[x])
            result.append(max_sum)

        return result