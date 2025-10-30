import bisect
from typing import List

class SegmentTree:
    def __init__(self, size):
        self.tree = [-1] * (4 * size)
    
    def update(self, idx, value, node=1, node_left=0, node_right=None):
        if node_right is None:
            node_right = self.size - 1
        if node_left == node_right:
            self.tree[node] = max(self.tree[node], value)
            return
        mid = (node_left + node_right) // 2
        if idx <= mid:
            self.update(idx, value, 2 * node, node_left, mid)
        else:
            self.update(idx, value, 2 * node + 1, mid + 1, node_right)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, left, right, node=1, node_left=0, node_right=None):
        if node_right is None:
            node_right = self.size - 1
        if left > node_right or right < node_left:
            return -1
        if left <= node_left and right >= node_right:
            return self.tree[node]
        mid = (node_left + node_right) // 2
        left_max = self.query(left, right, 2 * node, node_left, mid)
        right_max = self.query(left, right, 2 * node + 1, mid + 1, node_right)
        return max(left_max, right_max)

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Step 1: Collect unique nums2 values and sort them
        unique_nums2 = sorted(set(nums2))
        self.size = len(unique_nums2)
        discretization = {num: idx for idx, num in enumerate(unique_nums2)}
        
        # Step 2: Sort pairs (nums1[j], nums2[j]) in descending order of nums1[j]
        pairs = sorted(zip(nums1, nums2), key=lambda x: -x[0])
        
        # Step 3: Sort queries with their indices based on x_i in descending order
        sorted_queries = sorted(enumerate(queries), key=lambda x: -x[1][0])
        
        # Step 4: Initialize segment tree
        seg_tree = SegmentTree(self.size)
        
        # Step 5: Process queries in sorted order
        answer = [-1] * len(queries)
        pair_idx = 0
        for q_idx, (x, y) in sorted_queries:
            # Add pairs where nums1[j] >= x to the segment tree
            while pair_idx < len(pairs) and pairs[pair_idx][0] >= x:
                j_num1, j_num2 = pairs[pair_idx]
                discretized_j_num2 = discretization[j_num2]
                seg_tree.update(discretized_j_num2, j_num1 + j_num2)
                pair_idx += 1
            # Find the discretized value for y
            if y > unique_nums2[-1]:
                answer[q_idx] = -1
            else:
                left = bisect.bisect_left(unique_nums2, y)
                if left == self.size:
                    answer[q_idx] = -1
                else:
                    answer[q_idx] = seg_tree.query(left, self.size - 1)
        return answer