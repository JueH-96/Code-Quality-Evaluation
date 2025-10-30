from typing import List
import heapq

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        i, j = 0, 0
        q1, q2 = [], []
        while i < n or j < m:
            while i < n - 1 and nums1[i] == nums1[i + 1]:
                heapq.heappush(q1, nums1[i])
                i += 1
            while j < m - 1 and nums2[j] == nums2[j + 1]:
                heapq.heappush(q2, nums2[j])
                j += 1
            if i < n and j < m:
                if nums1[i] < nums2[j]:
                    heapq.heappush(q1, nums1[i])
                    i += 1
                elif nums1[i] > nums2[j]:
                    heapq.heappush(q2, nums2[j])
                    j += 1
                else:
                    i += 1
                    j += 1
            elif i < n:
                heapq.heappush(q1, nums1[i])
                i += 1
            else:
                heapq.heappush(q2, nums2[j])
                j += 1
        if len(q1) > len(q2):
            q1, q2 = q2, q1
        while len(q1) < len(q2):
            heapq.heappush(q1, heapq.heappop(q2))
        while q1 and q2 and heapq.heappop(q1) == heapq.heappop(q2):
            pass
        if not q1:
            return -1
        return sum(q1) + sum(q2)