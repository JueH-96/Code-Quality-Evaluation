import bisect
from typing import List

class SegmentTree:
    def __init__(self, size):
        # size: number of leaves in the segment tree.
        # This will be len(all_nums2_values), the number of distinct nums2 values.
        
        # Calculate the actual size of the tree array.
        # It's the smallest power of 2 greater than or equal to 'size'.
        # This makes segment tree operations cleaner (power-of-2 size).
        self.size = 1
        while self.size < size:
            self.size *= 2
        
        # Initialize the segment tree array.
        # Tree nodes will store the maximum sum found in their respective ranges.
        # Initialize with -1, as problem requires -1 if no valid sum is found,
        # and sums (nums1[j] + nums2[j]) are at least 1+1=2, so -1 is a safe sentinel.
        self.tree = [-1] * (2 * self.size)

    def update(self, index, value):
        # 'index' is the 0-based compressed coordinate for a nums2 value.
        # 'value' is the sum (nums1[j] + nums2[j]) to potentially update.
        
        # Map the 0-based index to the corresponding leaf node in the tree array.
        # Leaves are typically stored from 'self.size' to '2*self.size - 1'.
        idx = index + self.size
        
        # Update the leaf node: take the maximum of current value and new value.
        self.tree[idx] = max(self.tree[idx], value)
        
        # Propagate the maximum value up the tree to parent nodes.
        # A parent node's value is the maximum of its children.
        # We start from the leaf and go up to the root (index 1).
        while idx > 1:
            # Parent node is at idx // 2.
            # Its children are at idx (current node) and idx ^ 1 (its sibling).
            self.tree[idx // 2] = max(self.tree[idx], self.tree[idx ^ 1])
            idx //= 2 # Move to parent

    def query(self, left, right):
        # 'left' and 'right' are 0-based compressed coordinates for the query range [left, right].
        # Returns the maximum sum in this range.
        
        # Map the 0-based range to the corresponding indices in the tree array.
        left += self.size
        right += self.size
        
        res = -1 # Initialize result with -1
        
        # Iterate up the tree, covering the query range.
        # This iterative approach efficiently queries maximums in ranges.
        while left <= right:
            # If 'left' is a right child (odd index), it means its parent's range
            # does not fully cover the current 'left' segment. So, include 'left' node's value.
            if left % 2 == 1: 
                res = max(res, self.tree[left])
                left += 1 # Move to the next segment to the right (sibling)
            
            # If 'right' is a left child (even index), it means its parent's range
            # does not fully cover the current 'right' segment. So, include 'right' node's value.
            if right % 2 == 0:
                res = max(res, self.tree[right])
                right -= 1 # Move to the next segment to the left (sibling)
            
            # Move up to the parent level.
            left //= 2
            right //= 2
            
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Step 1: Create a list of data points.
        # Each point consists of (nums1_val, nums2_val, sum_val).
        # We sort these points by nums1_val in descending order.
        points = []
        for j in range(n):
            points.append((nums1[j], nums2[j], nums1[j] + nums2[j]))
        points.sort(key=lambda x: x[0], reverse=True)
        
        # Step 2: Create a list of queries, preserving their original indices.
        # We sort these queries by their x_i value in descending order.
        queries_with_idx = []
        for i, (x_i, y_i) in enumerate(queries):
            # The problem statement has a typo in description x_i == queries[i][1], y_i == queries[i][2]
            # but example clarifies it's [x_i, y_i]. So, use x_i = q[0], y_i = q[1].
            queries_with_idx.append((x_i, y_i, i))
        queries_with_idx.sort(key=lambda x: x[0], reverse=True)
        
        # Step 3: Prepare for coordinate compression for nums2 values.
        # We need to map actual nums2 values (up to 10^9) to smaller 0-based indices
        # to use them as indices in the segment tree.
        # Create a sorted list of unique nums2 values from all data points.
        all_nums2_values = sorted(list(set([p[1] for p in points])))
        
        # Step 4: Initialize the Segment Tree.
        # The segment tree will operate on the compressed indices of nums2 values.
        seg_tree = SegmentTree(len(all_nums2_values))
        
        # Initialize the answer array.
        answer = [-1] * len(queries)
        
        # 'ptr' will keep track of the current position in the sorted 'points' list.
        ptr = 0 
        
        # Step 5: Process queries.
        # Iterate through the sorted queries.
        for x_i, y_i, original_idx in queries_with_idx:
            # For the current query (x_i, y_i), add all points (p_j) to the segment tree
            # such that p_j.nums1 >= x_i.
            # Since both points and queries are sorted by their first coordinate in descending order,
            # points will be added to the segment tree monotonically.
            while ptr < n and points[ptr][0] >= x_i:
                current_nums2 = points[ptr][1]
                current_sum = points[ptr][2]
                
                # Find the compressed index for the current_nums2 value.
                idx_in_coords = bisect.bisect_left(all_nums2_values, current_nums2)
                
                # Update the segment tree with the sum at this compressed index.
                # The segment tree will ensure that for each compressed index,
                # it stores the maximum sum encountered so far.
                seg_tree.update(idx_in_coords, current_sum)
                ptr += 1
            
            # After adding all relevant points for the current x_i constraint,
            # query the segment tree for the maximum sum where nums2 >= y_i.
            # First, find the starting compressed index for the query (smallest nums2 value >= y_i).
            query_start_idx = bisect.bisect_left(all_nums2_values, y_i)
            
            # If query_start_idx is within the valid range of compressed coordinates,
            # perform the segment tree query.
            if query_start_idx < len(all_nums2_values):
                # Query the segment tree from query_start_idx to the end of the compressed range.
                res = seg_tree.query(query_start_idx, len(all_nums2_values) - 1)
                answer[original_idx] = res # Store the result in the original query's position
            # If query_start_idx is out of bounds, it means no nums2 value in our collected set
            # satisfies nums2 >= y_i, so answer remains -1 (as initialized).
                
        return answer