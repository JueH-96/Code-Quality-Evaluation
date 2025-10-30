from typing import List

class Solution:
  def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
    N = len(nums)
    
    # P[i] = 1 if nums[i] is a peak in the global nums array, 0 otherwise.
    # Elements at index 0 and N-1 cannot be peaks as they don't have two neighbors.
    P = [0] * N

    # Helper function to determine if nums[k] is a peak.
    # This function relies on the `nums` list in the outer scope, which may be modified by updates.
    def is_peak_at_idx(k: int) -> int:
        # A peak must be strictly between the first and last elements of the array.
        # So, k must be in the range [1, N-2] (or (0, N-1) using 0-based indexing).
        if not (0 < k < N - 1): # Equivalent to 1 <= k <= N - 2
            return 0
        if nums[k] > nums[k-1] and nums[k] > nums[k+1]:
            return 1
        return 0

    # Initialize P based on the initial state of nums.
    # Only indices from 1 to N-2 can potentially be peaks.
    for i in range(1, N - 1):
        P[i] = is_peak_at_idx(i)

    # Segment Tree implementation as a nested class.
    class SegmentTree:
        def __init__(self, size: int, initial_values: List[int]):
            self.n = size
            # The segment tree array typically needs about 4*n space.
            self.tree = [0] * (4 * self.n)
            # Build the tree from the initial_values (our P array).
            self._build(1, 0, self.n - 1, initial_values)

        def _build(self, v_node: int, tl_node: int, tr_node: int, initial_values: List[int]):
            # v_node: index of the current node in self.tree (1-indexed).
            # tl_node, tr_node: range [tl_node, tr_node] covered by this node.
            if tl_node == tr_node: # Leaf node
                self.tree[v_node] = initial_values[tl_node]
            else: # Internal node
                tm_node = (tl_node + tr_node) // 2 # Midpoint
                # Recursively build left and right children
                self._build(2 * v_node, tl_node, tm_node, initial_values)
                self._build(2 * v_node + 1, tm_node + 1, tr_node, initial_values)
                # Current node's value is sum of its children's values
                self.tree[v_node] = self.tree[2 * v_node] + self.tree[2 * v_node + 1]

        def _query(self, v_node: int, tl_node: int, tr_node: int, query_l: int, query_r: int) -> int:
            # Current segment [tl_node, tr_node] is completely outside query range [query_l, query_r].
            if tl_node > query_r or tr_node < query_l:
                return 0
            # Current segment is completely inside query range.
            if query_l <= tl_node and tr_node <= query_r:
                return self.tree[v_node]
            
            # Current segment partially overlaps with query range. Recurse on children.
            tm_node = (tl_node + tr_node) // 2
            sum_left = self._query(2 * v_node, tl_node, tm_node, query_l, query_r)
            sum_right = self._query(2 * v_node + 1, tm_node + 1, tr_node, query_l, query_r)
            return sum_left + sum_right

        def _update(self, v_node: int, tl_node: int, tr_node: int, pos: int, new_val: int):
            # Reached the leaf node corresponding to pos.
            if tl_node == tr_node:
                self.tree[v_node] = new_val
            else: # Internal node, decide whether to go left or right.
                tm_node = (tl_node + tr_node) // 2
                if pos <= tm_node: # pos is in the left child's range.
                    self._update(2 * v_node, tl_node, tm_node, pos, new_val)
                else: # pos is in the right child's range.
                    self._update(2 * v_node + 1, tm_node + 1, tr_node, pos, new_val)
                # Update current node's sum based on (potentially modified) children.
                self.tree[v_node] = self.tree[2 * v_node] + self.tree[2 * v_node + 1]
        
        # Public method for range sum query.
        def range_sum(self, ql: int, qr: int) -> int:
            if ql > qr: # Handle empty or invalid ranges.
                return 0
            return self._query(1, 0, self.n - 1, ql, qr)

        # Public method for point update.
        def point_update(self, pos: int, new_val: int):
            self._update(1, 0, self.n - 1, pos, new_val)
    
    # Instantiate the segment tree.
    seg_tree = SegmentTree(N, P)

    results = [] # To store answers for type 1 queries.
    for query_item in queries:
        query_type = query_item[0]
        
        if query_type == 1: # Count peaks query
            l, r = query_item[1], query_item[2]
            # Peaks in nums[l..r] must be at an index k where l < k < r.
            # This means we query the sum of P[k] for k from l+1 to r-1.
            count = seg_tree.range_sum(l + 1, r - 1)
            results.append(count)
        
        else: # query_type == 2: Update element query
            idx, val = query_item[1], query_item[2]

            # Optimization: If the value at nums[idx] doesn't actually change,
            # no peak statuses can change. Skip updates.
            if nums[idx] == val:
                continue

            # Perform the update in the nums array.
            nums[idx] = val 
            
            # Re-evaluate peak status for indices whose peak status might have changed.
            # These are: idx itself, and its neighbors idx-1 and idx+1.
            # We only care about them if they are valid indices for being a peak (i.e., 1 to N-2).
            indices_to_recheck = set()
            for k_candidate in [idx - 1, idx, idx + 1]:
                if 1 <= k_candidate <= N - 2: # Check if k_candidate is a potential peak index.
                    indices_to_recheck.add(k_candidate)
            
            for k_check in indices_to_recheck:
                # Calculate new peak status using the updated nums array.
                new_peak_status_at_k = is_peak_at_idx(k_check)
                
                # P[k_check] currently holds the old peak status.
                # If it changed, update segment tree and P array.
                if new_peak_status_at_k != P[k_check]:
                    seg_tree.point_update(k_check, new_peak_status_at_k)
                    P[k_check] = new_peak_status_at_k # Keep P array in sync.
        
    return results