import bisect
from typing import List

class SegmentTree:
    def __init__(self, size: int):
        self.size = size
        self.tree = [-1] * (4 * size)

    def update(self, pos: int, val: int, node: int = 1, start: int = 0, end: int = None):
        if end is None:
            end = self.size - 1
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if pos <= mid:
                self.update(pos, val, node * 2, start, mid)
            else:
                self.update(pos, val, node * 2 + 1, mid + 1, end)
            self.tree[node] = max(self.tree[node * 2], self.tree[node * 2 + 1])

    def get_val(self, pos: int, node: int = 1, start: int = 0, end: int = None):
        if end is None:
            end = self.size - 1
        if start == end:
            return self.tree[node]
        mid = (start + end) // 2
        if pos <= mid:
            return self.get_val(pos, node * 2, start, mid)
        else:
            return self.get_val(pos, node * 2 + 1, mid + 1, end)

    def query(self, left: int, right: int, node: int = 1, start: int = 0, end: int = None):
        if end is None:
            end = self.size - 1
        if left > right or left > end or right < start:
            return -1
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        left_max = self.query(left, right, node * 2, start, mid)
        right_max = self.query(left, right, node * 2 + 1, mid + 1, end)
        return max(left_max, right_max)

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        b_unique_asc = sorted(set(nums2))
        m = len(b_unique_asc)
        
        # Precompute points with a, b_rank, sum_ab
        points = []
        for num1, num2 in zip(nums1, nums2):
            b_rank = bisect.bisect_left(b_unique_asc, num2)
            sum_ab = num1 + num2
            points.append((num1, b_rank, sum_ab))
        
        # Sort points by a decreasing
        points.sort(key=lambda x: x[0], reverse=True)
        
        # Prepare and sort queries with original index by x decreasing
        query_list = [(x, y, idx) for idx, (x, y) in enumerate(queries)]
        query_list.sort(key=lambda x: x[0], reverse=True)
        
        # Initialize segment tree
        seg_tree = SegmentTree(m)
        
        # Initialize point index
        idx_point = 0
        
        # Answer list
        answer = [0] * len(queries)
        
        # Process each query in sorted order
        for x_q, y_q, orig_idx in query_list:
            # Add all points with a >= x_q
            while idx_point < n and points[idx_point][0] >= x_q:
                a_pt, b_rank_pt, sum_ab_pt = points[idx_point]
                curr_val = seg_tree.get_val(b_rank_pt)
                new_val = max(curr_val, sum_ab_pt)
                seg_tree.update(b_rank_pt, new_val)
                idx_point += 1
            
            # Handle the query with y_q
            low_rank = bisect.bisect_left(b_unique_asc, y_q)
            if low_rank >= m:
                ans_val = -1
            else:
                ans_val = seg_tree.query(low_rank, m - 1)
            answer[orig_idx] = ans_val
        
        return answer