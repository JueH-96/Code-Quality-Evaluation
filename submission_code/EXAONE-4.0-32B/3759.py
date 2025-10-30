import heapq
from typing import List

class MaxKSum:
    def __init__(self, k):
        self.k = k
        self.heap = []
        self.total = 0
        
    def insert(self, x):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, x)
            self.total += x
        else:
            if self.heap and x > self.heap[0]:
                popped = heapq.heappop(self.heap)
                self.total -= popped
                heapq.heappush(self.heap, x)
                self.total += x
                
    def query(self):
        return self.total

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        indices = list(range(n))
        indices.sort(key=lambda i: (nums1[i], i))
        
        ds = MaxKSum(k)
        ans = [0] * n
        i = 0
        while i < n:
            j = i
            current_val = nums1[indices[i]]
            while j < n and nums1[indices[j]] == current_val:
                j += 1
            for idx_in_group in range(i, j):
                orig_idx = indices[idx_in_group]
                ans[orig_idx] = ds.query()
            for idx_in_group in range(i, j):
                orig_idx = indices[idx_in_group]
                ds.insert(nums2[orig_idx])
            i = j
            
        return ans