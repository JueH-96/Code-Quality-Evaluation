from typing import List
import heapq

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        m = len(queries)
        marked = [0] * n
        unmarked = []
        for i in range(n):
            unmarked.append((nums[i], i))
        unmarked.sort()
        j = 0
        total = sum(nums)
        res = []
        for i in range(m):
            index, k = queries[i]
            while j < n and (marked[unmarked[j][1]] or unmarked[j][1] == index):
                if marked[unmarked[j][1]] == 0:
                    total -= unmarked[j][0]
                j += 1
            marked[index] = 1
            total -= nums[index]
            while j < n and k > 0:
                if marked[unmarked[j][1]] == 0:
                    total -= unmarked[j][0]
                    marked[unmarked[j][1]] = 1
                    k -= 1
                j += 1
            res.append(total)
        return res