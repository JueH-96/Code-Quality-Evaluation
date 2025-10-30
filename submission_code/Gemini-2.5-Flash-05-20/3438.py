import math
from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        N = len(nums)
        
        # Segment Tree array. Using 4*N as a safe size for the tree nodes.
        # Tree nodes are 1-indexed for convenience in segment tree common implementations.
        tree = [0] * (4 * N) 
        
        # Helper function to determine if nums[k] is a peak
        # Returns 1 if peak, 0 otherwise.
        # A peak must be strictly greater than its neighbors and not at the array boundaries.
        def _is_peak(k: int) -> int:
            # First and last elements (indices 0 and N-1) cannot be peaks.
            # Also, ensure k-1 and k+1 are valid indices.
            if k <= 0 or k >= N - 1:
                return 0 
            
            # Check if nums[k] is greater than both its left and right neighbors
            if nums[k] > nums[k-1] and nums[k] > nums[k+1]:
                return 1
            return 0

        # Build function for the segment tree
        # node: current node index in the `tree` array
        # start: starting index of the segment in `nums` that this node covers
        # end: ending index of the segment in `nums` that this node covers
        def build(node: int, start: int, end: int):
            if start == end:
                # This is a leaf node, representing a single element from `nums`.
                # Its value is the peak status of `nums[start]`.
                tree[node] = _is_peak(start)
            else:
                # This is an internal node. Recursively build its children.
                mid = (start + end) // 2
                build(2 * node, start, mid)         # Build left child
                build(2 * node + 1, mid + 1, end)   # Build right child
                # The current node's value is the sum of its children's values.
                # This represents the total count of peaks in its covered range.
                tree[node] = tree[2 * node] + tree[2 * node + 1]

        # Update function for the segment tree (point update)
        # idx: the index in `nums` whose peak status needs to be updated
        # val: the new peak status (1 for peak, 0 for not peak) for `nums[idx]`
        def update(node: int, start: int, end: int, idx: int, val: int):
            if start == end:
                # Found the leaf node corresponding to `idx`, update its value.
                tree[node] = val
            else:
                # Traverse down to find the correct leaf node.
                mid = (start + end) // 2
                if start <= idx <= mid:
                    # `idx` is in the left child's range
                    update(2 * node, start, mid, idx, val)
                else:
                    # `idx` is in the right child's range
                    update(2 * node + 1, mid + 1, end, idx, val)
                # After updating a child, update the current node's sum based on its children.
                tree[node] = tree[2 * node] + tree[2 * node + 1]

        # Query function for the segment tree (range sum query)
        # qL, qR: the query range [qL, qR] (inclusive) for which to sum peak statuses
        def query(node: int, start: int, end: int, qL: int, qR: int) -> int:
            # Case 1: The current node's range is completely outside the query range.
            # No overlap, so it contributes 0 to the sum.
            if qR < start or end < qL:
                return 0
            
            # Case 2: The current node's range is completely within the query range.
            # All elements in this node's range contribute to the query, so return its sum.
            if qL <= start and end <= qR:
                return tree[node]
            
            # Case 3: The current node's range partially overlaps with the query range.
            # Recursively query left and right children and sum their results.
            mid = (start + end) // 2
            p1 = query(2 * node, start, mid, qL, qR)         # Query left child
            p2 = query(2 * node + 1, mid + 1, end, qL, qR)   # Query right child
            return p1 + p2

        # --- Main Logic ---

        # 1. Initialize the segment tree based on the initial peak statuses of `nums`.
        build(1, 0, N - 1)

        results = [] # To store the answers for type 1 queries

        for q in queries:
            q_type = q[0]

            if q_type == 1: # Count peaks in subarray nums[l..r]
                l, r = q[1], q[2]
                
                # According to the problem, a peak in subarray nums[l..r] must be at index i
                # such that l < i < r. This means the indices to check are from l+1 to r-1.
                query_left_idx = l + 1
                query_right_idx = r - 1
                
                # If the calculated query range for peaks is invalid (e.g., l+1 > r-1),
                # it means there are no possible peak elements in the specified subarray.
                if query_left_idx > query_right_idx:
                    results.append(0)
                else:
                    # Perform a range sum query on the segment tree for the valid peak indices.
                    count = query(1, 0, N - 1, query_left_idx, query_right_idx)
                    results.append(count)

            elif q_type == 2: # Change nums[index] to val
                idx, val = q[1], q[2]
                
                # An update to `nums[idx]` can potentially change the peak status of:
                # `nums[idx-1]`, `nums[idx]`, and `nums[idx+1]`.
                # We collect these indices to re-evaluate them.
                indices_to_recheck = set()
                
                # Add `idx-1` if it's a valid index
                if idx - 1 >= 0:
                    indices_to_recheck.add(idx - 1)
                
                # Add `idx` (always valid)
                indices_to_recheck.add(idx) 
                
                # Add `idx+1` if it's a valid index
                if idx + 1 < N:
                    indices_to_recheck.add(idx + 1)
                
                # Store the current peak statuses of these potentially affected indices
                # BEFORE updating `nums[idx]`.
                old_peak_statuses = {}
                for k in indices_to_recheck:
                    old_peak_statuses[k] = _is_peak(k)

                # Update the value in the global `nums` array
                nums[idx] = val

                # Re-evaluate the peak statuses for the collected indices.
                # If a status has changed, update the segment tree.
                for k in indices_to_recheck:
                    new_peak_status = _is_peak(k)
                    if new_peak_status != old_peak_statuses[k]:
                        update(1, 0, N - 1, k, new_peak_status)
                        
        return results