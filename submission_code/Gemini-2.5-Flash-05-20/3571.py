import collections
import math
from typing import List

class SegmentTree:
    def __init__(self, size):
        # Size of the segment tree (number of leaves)
        # Find the smallest power of 2 greater than or equal to size
        self.size = 1
        while self.size < size:
            self.size *= 2
        
        # Tree array, initialized to 0 for maximum queries
        self.tree = [0] * (2 * self.size)

    def update(self, idx, val):
        # Update point at original index idx with value val
        # For RMQ, we take max(current_val, new_val)
        idx += self.size # Move to leaf node in the segment tree array
        self.tree[idx] = max(self.tree[idx], val)
        
        # Propagate changes up to the root
        while idx > 1:
            self.tree[idx // 2] = max(self.tree[idx], self.tree[idx ^ 1]) # idx ^ 1 gets sibling
            idx //= 2

    def query(self, l, r):
        # Query maximum in range [l, r] (inclusive)
        # If l > r, return 0 (identity for max, meaning an empty range has no elements to contribute to max)
        if l > r:
            return 0
        
        l += self.size # Move to leaf node
        r += self.size # Move to leaf node
        
        res = 0 # Initialize result with identity value (0 for max)
        
        while l <= r:
            if l % 2 == 1: # l is a right child, so its parent's range does not fully cover [l, r]
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0: # r is a left child, same logic as above
                res = max(res, self.tree[r])
                r -= 1
            l //= 2
            r //= 2
            
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        
        # Step 1: Coordinate Compression for y-coordinates
        all_y_coords = [p[1] for p in coordinates]
        unique_y_coords = sorted(list(set(all_y_coords)))
        y_to_compressed = {y: i for i, y in enumerate(unique_y_coords)}
        num_compressed_y = len(unique_y_coords)
        
        # Store points along with their original indices
        points_with_original_idx = []
        for i in range(n):
            points_with_original_idx.append((coordinates[i][0], coordinates[i][1], i))
            
        # Step 2: Calculate dp_prefix (max length of increasing path ending at coordinates[i])
        dp_prefix = [0] * n
        
        # Sort points by x then y for dp_prefix calculation.
        # If x_i == x_j, then y_i < y_j implies (x_i, y_i) cannot be before (x_j, y_j)
        # in an increasing path. The group-wise update handles this correctly.
        sorted_points_asc = sorted(points_with_original_idx)
        
        seg_tree_prefix = SegmentTree(num_compressed_y)
        
        # Use a sentinel value for current_x to correctly handle the first group of points.
        current_x = -float('inf') 
        group_updates = [] # Stores (compressed_y, calculated_len) for points with the same x-coordinate
        
        for x, y, original_idx in sorted_points_asc:
            if x != current_x:
                # Process updates for the previous x-coordinate group.
                # This is crucial: updates for points with the same X are only
                # applied to the segment tree AFTER all points in that X-group
                # have queried, ensuring they don't use each other's path lengths.
                for cy, length in group_updates:
                    seg_tree_prefix.update(cy, length)
                group_updates = []
                current_x = x

            compressed_y = y_to_compressed[y]
            # Query for max path length ending at a point (x_prev, y_prev) where x_prev < x and y_prev < y
            # y_prev_compressed corresponds to indices up to compressed_y - 1
            max_len_prev = seg_tree_prefix.query(0, compressed_y - 1)
            
            current_len = 1 + max_len_prev
            dp_prefix[original_idx] = current_len
            group_updates.append((compressed_y, current_len))
        
        # Process updates for the last x-coordinate group after the loop finishes
        for cy, length in group_updates:
            seg_tree_prefix.update(cy, length)
            
        # Step 3: Calculate dp_suffix (max length of increasing path starting at coordinates[i])
        dp_suffix = [0] * n
        
        # Sort points by x (descending) then y (descending) for dp_suffix calculation.
        # Similar group processing logic as dp_prefix.
        sorted_points_desc = sorted(points_with_original_idx, key=lambda p: (-p[0], -p[1]))
        
        seg_tree_suffix = SegmentTree(num_compressed_y)
        
        # Use a sentinel value for current_x to correctly handle the first group.
        current_x = float('inf') 
        group_updates = []
        
        for x, y, original_idx in sorted_points_desc:
            if x != current_x:
                # Process updates for the previous x-coordinate group
                for cy, length in group_updates:
                    seg_tree_suffix.update(cy, length)
                group_updates = []
                current_x = x

            compressed_y = y_to_compressed[y]
            # Query for max path length starting at a point (x_next, y_next) where x_next > x and y_next > y
            # y_next_compressed corresponds to indices from compressed_y + 1 to num_compressed_y - 1
            max_len_next = seg_tree_suffix.query(compressed_y + 1, num_compressed_y - 1)
            
            current_len = 1 + max_len_next
            dp_suffix[original_idx] = current_len
            group_updates.append((compressed_y, current_len))
            
        # Process updates for the last x-coordinate group
        for cy, length in group_updates:
            seg_tree_suffix.update(cy, length)
            
        # Step 4: Calculate final result for coordinates[k]
        # The target point coordinates[k] is counted in both dp_prefix[k] and dp_suffix[k],
        # so we subtract 1 to avoid double counting.
        return dp_prefix[k] + dp_suffix[k] - 1