import sys
from collections import defaultdict
from typing import List

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [ (0, -sys.maxsize, -sys.maxsize, -sys.maxsize) ] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = (data[i], data[i], data[i], data[i])
        for i in range(self.n, self.size):
            self.tree[self.size + i] = (0, -sys.maxsize, -sys.maxsize, -sys.maxsize)
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.merge(self.tree[2*i], self.tree[2*i+1])
    
    def merge(self, a, b):
        sum_ = a[0] + b[0]
        max_prefix = max(a[1], a[0] + b[1])
        max_suffix = max(b[2], b[0] + a[2])
        max_subarray = max(a[3], b[3], a[2] + b[1])
        return (sum_, max_prefix, max_suffix, max_subarray)
    
    def query(self, l, r):
        res_l = (0, -sys.maxsize, -sys.maxsize, -sys.maxsize)
        res_r = (0, -sys.maxsize, -sys.maxsize, -sys.maxsize)
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res_l = self.merge(res_l, self.tree[l])
                l += 1
            if r % 2 == 0:
                res_r = self.merge(self.tree[r], res_r)
                r -= 1
            l >>= 1
            r >>= 1
        return self.merge(res_l, res_r)

def kadane(nums):
    max_sum = -sys.maxsize
    current_sum = 0
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        if not nums:
            return 0
        original_max = kadane(nums)
        max_total = original_max
        
        occurrences = defaultdict(list)
        for idx, num in enumerate(nums):
            occurrences[num].append(idx)
        
        st = SegmentTree(nums)
        
        for x in occurrences:
            pos_list = occurrences[x]
            segments = []
            start = 0
            for pos in pos_list:
                end = pos - 1
                if start <= end:
                    sum_seg, max_prefix, max_suffix, max_sub = st.query(start, end)
                    segments.append((sum_seg, max_prefix, max_suffix, max_sub))
                start = pos + 1
            if start < len(nums):
                sum_seg, max_prefix, max_suffix, max_sub = st.query(start, len(nums) - 1)
                segments.append((sum_seg, max_prefix, max_suffix, max_sub))
            
            current_max_suffix = -sys.maxsize
            current_global = -sys.maxsize
            for seg in segments:
                sum_s, prefix_s, suffix_s, sub_s = seg
                option1 = suffix_s
                option2 = current_max_suffix + sum_s if current_max_suffix != -sys.maxsize else -sys.maxsize
                current_max_suffix = max(option1, option2)
                current_max = max(current_max_suffix, sub_s)
                if current_max > current_global:
                    current_global = current_max
            
            if current_global > max_total:
                max_total = current_global
        
        return max_total