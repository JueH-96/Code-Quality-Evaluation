from typing import List

class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        q = len(queries)
        
        # Step 1: Calculate coverage using difference array
        coverage_diff = [0] * (n + 1)
        for l, r in queries:
            coverage_diff[l] += 1
            coverage_diff[r + 1] -= 1
        
        coverage = [0] * n
        current = 0
        for i in range(n):
            current += coverage_diff[i]
            coverage[i] = current
        
        # Step 2: Calculate slack and check feasibility
        slack = [coverage[i] - nums[i] for i in range(n)]
        if any(s < 0 for s in slack):
            return -1
        
        # Step 3: Sort queries by end index
        sorted_queries = sorted(queries, key=lambda x: x[1])
        
        # Step 4: Implement Segment Tree with range minimum query and range update
        class SegmentTree:
            def __init__(self, data):
                from math import ceil, log2
                self.n = len(data)
                h = ceil(log2(self.n)) if self.n > 0 else 0
                self.size = 1 << (h + 1)
                self.min = [0] * self.size
                self.lazy = [0] * self.size
                self.build(data, 1, 0, self.n - 1)
            
            def build(self, data, node, l, r):
                if l == r:
                    self.min[node] = data[l]
                else:
                    mid = (l + r) // 2
                    self.build(data, 2*node, l, mid)
                    self.build(data, 2*node+1, mid+1, r)
                    self.min[node] = min(self.min[2*node], self.min[2*node+1])
            
            def push(self, node, l, r):
                if self.lazy[node] != 0:
                    self.min[node] += self.lazy[node]
                    if l != r:
                        self.lazy[2*node] += self.lazy[node]
                        self.lazy[2*node+1] += self.lazy[node]
                    self.lazy[node] = 0
            
            def range_min(self, node, l, r, ql, qr):
                self.push(node, l, r)
                if qr < l or ql > r:
                    return float('inf')
                if ql <= l and r <= qr:
                    return self.min[node]
                mid = (l + r) // 2
                return min(self.range_min(2*node, l, mid, ql, qr),
                           self.range_min(2*node+1, mid+1, r, ql, qr))
            
            def range_add(self, node, l, r, ql, qr, val):
                self.push(node, l, r)
                if qr < l or ql > r:
                    return
                if ql <= l and r <= qr:
                    self.lazy[node] += val
                    self.push(node, l, r)
                    return
                mid = (l + r) // 2
                self.range_add(2*node, l, mid, ql, qr, val)
                self.range_add(2*node+1, mid+1, r, ql, qr, val)
                self.min[node] = min(self.min[2*node], self.min[2*node+1])
        
        # Initialize the segment tree
        seg_tree = SegmentTree(slack)
        
        removed = 0
        for l, r in sorted_queries:
            min_slack = seg_tree.range_min(1, 0, n-1, l, r)
            if min_slack >= 1:
                # Remove this query by decrementing slack in [l, r]
                seg_tree.range_add(1, 0, n-1, l, r, -1)
                removed += 1
        
        return removed