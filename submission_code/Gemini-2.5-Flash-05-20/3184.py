import collections
import bisect
from typing import List

class Solution:
    # Nested SegmentTree class to manage range maximum queries
    class SegmentTree:
        def __init__(self, size: int, default_value: int = -float('inf')):
            self.size = size
            # The tree array uses 4 * size for complete binary tree representation
            # This is a common heuristic to prevent index out of bounds for arbitrary tree shapes.
            self.tree = [default_value] * (4 * size)
            self.default_value = default_value

        def _update_tree(self, node: int, start: int, end: int, idx: int, val: int):
            """
            Internal recursive helper for updating a value at a specific index.
            `idx` is the compressed coordinate where `val` should be updated.
            The update logic is `max(current_value, val)` at the leaf node.
            """
            if start == end: # Base case: Leaf node reached
                self.tree[node] = max(self.tree[node], val)
                return

            mid = (start + end) // 2
            if start <= idx <= mid:
                # Target index is in the left child's range
                self._update_tree(2 * node + 1, start, mid, idx, val)
            else:
                # Target index is in the right child's range
                self._update_tree(2 * node + 2, mid + 1, end, idx, val)
            
            # Update parent node: it stores the maximum of its children
            self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])

        def update(self, idx: int, val: int):
            """
            Public method to update the segment tree.
            `idx` is the compressed coordinate.
            `val` is the new dp value associated with this coordinate.
            """
            self._update_tree(0, 0, self.size - 1, idx, val)

        def _query_tree(self, node: int, start: int, end: int, l: int, r: int):
            """
            Internal recursive helper for querying the maximum value in a given range [l, r].
            `l` and `r` are compressed coordinates.
            """
            # Case 1: Current segment is completely outside the query range
            if r < start or end < l:
                return self.default_value
            
            # Case 2: Current segment is completely inside the query range
            if l <= start and end <= r:
                return self.tree[node]
            
            # Case 3: Current segment partially overlaps with the query range
            mid = (start + end) // 2
            p1 = self._query_tree(2 * node + 1, start, mid, l, r)
            p2 = self._query_tree(2 * node + 2, mid + 1, end, l, r)
            return max(p1, p2)

        def query(self, l: int, r: int):
            """
            Public method to query the segment tree for the maximum value in range [l, r].
            """
            # Handle invalid query range (e.g., l > r), though for this problem, l will always be 0.
            if l > r: 
                return self.default_value
            return self._query_tree(0, 0, self.size - 1, l, r)

    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Calculate val_k = nums[k] - k for all k.
        # These are the values that define the non-decreasing property for a balanced subsequence.
        val_minus_idx = [nums[i] - i for i in range(n)]
        
        # Step 2: Perform coordinate compression.
        # Get unique values from val_minus_idx and sort them.
        # This `coords` list will serve as the mapping for segment tree indices.
        coords = sorted(list(set(val_minus_idx)))
        
        # Step 3: Initialize the Segment Tree.
        # The size of the segment tree corresponds to the number of unique coordinate values.
        m = len(coords)
        seg_tree = self.SegmentTree(m) # Instantiate the nested SegmentTree class

        # Step 4: Initialize the overall maximum sum.
        # A single element subsequence (e.g., [nums[i]]) is always balanced.
        # Therefore, the maximum sum must be at least the largest element in `nums`.
        # This correctly handles cases where all numbers are negative, or when the optimal
        # subsequence consists of just one element.
        max_overall_sum = max(nums) 

        # Step 5: Iterate through `nums` to compute DP values and update the Segment Tree.
        for i in range(n):
            current_val_minus_idx = nums[i] - i
            
            # Find the compressed index for `current_val_minus_idx`.
            # `bisect_left` returns the appropriate index in `coords`, which is our compressed index.
            compressed_idx = bisect.bisect_left(coords, current_val_minus_idx)
            
            # Query the segment tree for the maximum dp value in the range [0, compressed_idx].
            # This query finds `max(dp[j])` for all `j < i` where `(nums[j]-j) <= (nums[i]-i)`.
            prev_max_dp = seg_tree.query(0, compressed_idx)

            # Calculate `dp_i_val`: the maximum sum of a balanced subsequence ending at index `i`.
            # Option 1: `nums[i]` itself forms a balanced subsequence of length 1.
            dp_i_val = nums[i]
            
            # Option 2: Extend a previously found balanced subsequence.
            # Only consider extending if `prev_max_dp` is not the default negative infinity,
            # meaning a valid previous subsequence was found.
            if prev_max_dp != seg_tree.default_value:
                dp_i_val = max(dp_i_val, nums[i] + prev_max_dp)
            
            # Update the overall maximum sum found so far.
            max_overall_sum = max(max_overall_sum, dp_i_val)
            
            # Update the segment tree at `compressed_idx` with `dp_i_val`.
            # The `update` method internally handles storing the maximum value if multiple
            # `(nums[k]-k)` values map to the same `compressed_idx`.
            seg_tree.update(compressed_idx, dp_i_val)
            
        return max_overall_sum