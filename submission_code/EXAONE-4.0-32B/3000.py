import math
from typing import List

class SegmentTree:
    def __init__(self, m):
        self.m = m
        self.size = 1
        while self.size < m:
            self.size *= 2
        self.max_tree = [-1] * (2 * self.size)
        self.min_tree = [self.m] * (2 * self.size)
    
    def activate(self, idx):
        i = idx + self.size
        if self.max_tree[i] == -1:
            self.max_tree[i] = idx
            self.min_tree[i] = idx
            i //= 2
            while i:
                left_child = 2 * i
                right_child = 2 * i + 1
                self.max_tree[i] = max(self.max_tree[left_child], self.max_tree[right_child])
                self.min_tree[i] = min(self.min_tree[left_child], self.min_tree[right_child])
                i //= 2
                
    def query_max(self, l, r):
        l0 = l + self.size
        r0 = r + self.size
        res = -1
        while l0 <= r0:
            if l0 % 2 == 1:
                if self.max_tree[l0] > res:
                    res = self.max_tree[l0]
                l0 += 1
            if r0 % 2 == 0:
                if self.max_tree[r0] > res:
                    res = self.max_tree[r0]
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

    def query_min(self, l, r):
        l0 = l + self.size
        r0 = r + self.size
        res = self.m
        while l0 <= r0:
            if l0 % 2 == 1:
                if self.min_tree[l0] < res:
                    res = self.min_tree[l0]
                l0 += 1
            if r0 % 2 == 0:
                if self.min_tree[r0] < res:
                    res = self.min_tree[r0]
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        sorted_vals = sorted(set(nums))
        comp = {val: idx for idx, val in enumerate(sorted_vals)}
        m = len(sorted_vals)
        seg_tree = SegmentTree(m)
        min_diff = 10**18
        
        for j in range(n):
            if j >= x:
                idx_act = comp[nums[j - x]]
                seg_tree.activate(idx_act)
            if j >= x:
                c = comp[nums[j]]
                idx1 = seg_tree.query_max(0, c)
                if idx1 == -1:
                    diff1 = 10**18
                else:
                    diff1 = abs(nums[j] - sorted_vals[idx1])
                idx2 = seg_tree.query_min(c, m - 1)
                if idx2 == m:
                    diff2 = 10**18
                else:
                    diff2 = abs(nums[j] - sorted_vals[idx2])
                min_diff = min(min_diff, diff1, diff2)
                
        return min_diff