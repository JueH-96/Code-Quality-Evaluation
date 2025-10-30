import heapq
from itertools import groupby
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        indices = list(range(n))
        indices.sort(key=lambda i: nums1[i])
        res = [0] * n
        heap = []
        s = 0
        
        for key, group_iter in groupby(indices, key=lambda i: nums1[i]):
            group = list(group_iter)
            for i in group:
                res[i] = s
            for i in group:
                x = nums2[i]
                if len(heap) < k:
                    heapq.heappush(heap, x)
                    s += x
                else:
                    if x > heap[0]:
                        removed = heapq.heappushpop(heap, x)
                        s += (x - removed)
        return res