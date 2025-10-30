import heapq
import collections
import bisect
import math
from typing import List

# Segment Tree class optimized for range minimum queries and point updates.
# Uses 1-based indexing for tree nodes internally.
# Handles array indexing (0-based) correctly in public methods.
class SegmentTree:
    """
    Segment Tree implementation supporting Range Minimum Query and Point Update.
    Initialized with a size, can be built from an array.
    Uses float('inf') as the identity element for minimum operation.
    """
    def __init__(self, size):
        # Ensure size is non-negative
        if size < 0:
             raise ValueError("Size cannot be negative")
        self.n = size
        # Initialize tree with infinity. Size 4*n guarantees enough space.
        self.tree = [float('inf')] * (4 * self.n)

    # Internal recursive function to build the segment tree from an array `arr`
    def _build(self, arr, node, start, end):
        # Base case: leaf node
        if start == end:
            # Assign value from array if index is valid
            if 0 <= start < len(arr):
                 self.tree[node] = arr[start]
            # else leave it as inf (initial value)
        else:
            mid = (start + end) // 2
            # Recursively build left and right children
            self._build(arr, 2 * node, start, mid)
            self._build(arr, 2 * node + 1, mid + 1, end)
            # Internal node stores the minimum of its children
            # Ensure child nodes exist before accessing (should be safe with 4*n size)
            left_child_val = self.tree[2 * node] if 2 * node < len(self.tree) else float('inf')
            right_child_val = self.tree[2 * node + 1] if 2 * node + 1 < len(self.tree) else float('inf')
            self.tree[node] = min(left_child_val, right_child_val)


    # Public method to build the tree from a given array `arr`
    def build_from_array(self, arr):
        """ Builds the segment tree from the given array `arr`. """
        # Check if size matches expectation
        if len(arr) != self.n:
             # Option: Adjust self.n or raise error. Let's assume caller ensures match.
             # If self.n == 0, do nothing.
             if self.n == 0: return
             # Consider raising error if mismatch is critical
             # raise ValueError("Array size does not match SegmentTree size") 
             # For this problem, len(arr) == self.n is expected based on unique capacities count.
        
        # Start the recursive build process from the root node (index 1)
        # Check self.n > 0 before building to avoid issues with empty range
        if self.n > 0:
            self._build(arr, 1, 0, self.n - 1)

    # Internal recursive function for range minimum query on range [l, r]
    def _query(self, node, start, end, l, r):
        # If the current node's range [start, end] is completely outside the query range [l, r]
        if r < start or end < l or start > end: 
            return float('inf') # Return identity element for min operation
        
        # If the current node's range is completely within the query range [l, r]
        if l <= start and end <= r:
            # Ensure node index is valid before access (defensive check)
            return self.tree[node] if node < len(self.tree) else float('inf')
        
        # If the current node's range partially overlaps with the query range
        mid = (start + end) // 2
        # Recursively query the left and right children
        p1 = self._query(2 * node, start, mid, l, r)
        p2 = self._query(2 * node + 1, mid + 1, end, l, r)
        # Return the minimum of the results from children
        return min(p1, p2)
    
    # Public method for range minimum query
    def query_range(self, l, r):
        """ Queries the minimum value in the range [l, r]. """
        # Handle empty tree case
        if self.n == 0: return float('inf') 
        # Validate and clamp query range [l, r] to be within [0, n-1]
        l = max(0, l)
        r = min(self.n - 1, r)
        # If the validated range is invalid (e.g., l > r after clamping)
        if l > r: return float('inf')
        
        # Start the recursive query from the root node (index 1)
        return self._query(1, 0, self.n - 1, l, r)

    # Internal recursive function for point update at index `idx` with value `val`
    def _update(self, node, start, end, idx, val):
        # If the target index `idx` is outside the current node's range [start, end]
        if start > end or idx < start or idx > end:
             return # Index not in this subtree

        # Base case: reached the leaf node corresponding to the index `idx`
        if start == end:
            # Check if this node corresponds to the target index idx
            if start == idx:
                # Ensure node index is valid before updating (defensive check)
                 if node < len(self.tree):
                    self.tree[node] = val
            return # Stop recursion

        # If not a leaf node, decide whether to recurse into the left or right child
        mid = (start + end) // 2
        if idx <= mid: # Index is in the left child's range
            self._update(2 * node, start, mid, idx, val)
        else: # Index is in the right child's range
            self._update(2 * node + 1, mid + 1, end, idx, val)
        
        # After updating a child, update the current node's value based on its children
        # Ensure child node indices are valid before access (defensive check)
        left_child_val = self.tree[2 * node] if 2 * node < len(self.tree) else float('inf')
        right_child_val = self.tree[2 * node + 1] if 2 * node + 1 < len(self.tree) else float('inf')
        if node < len(self.tree):
            self.tree[node] = min(left_child_val, right_child_val)

    # Public method for point update
    def update_value(self, idx, val):
        """ Updates the value at index `idx` to `val`. """
        # Handle empty tree case
        if self.n == 0: return 
        # Validate index `idx` to be within [0, n-1]
        if idx < 0 or idx >= self.n: return # Index out of bounds
         
        # Start the recursive update from the root node (index 1)
        self._update(1, 0, self.n - 1, idx, val)


class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        """
        Calculates the number of unplaced fruit types based on the given rules.

        Args:
            fruits: A list of integers representing quantities of fruit types.
            baskets: A list of integers representing capacities of baskets.

        Returns:
            The number of fruit types that remain unplaced.
        """
        
        n = len(fruits)
        unplaced_count = 0
        
        # Step 1: Group basket indices by capacity using min-heaps.
        # This allows efficient retrieval of the minimum index for a given capacity.
        capacity_map = collections.defaultdict(list)
        for j in range(n):
            heapq.heappush(capacity_map[baskets[j]], j)
            
        # Step 2: Get a sorted list of unique basket capacities.
        unique_capacities = sorted(list(capacity_map.keys()))
        m = len(unique_capacities)
        
        # If there are no baskets (m=0), then all fruits are unplaced.
        # Note: Constraints state n >= 1, so baskets list is non-empty.
        # This check handles edge cases, though perhaps unnecessary under constraints.
        if m == 0: 
             return n 

        # Step 3: Create a mapping from capacity to its index in the sorted unique_capacities list.
        # This is needed to map capacities to leaf indices in the segment tree.
        capacity_to_idx = {cap: i for i, cap in enumerate(unique_capacities)}
        
        # Step 4: Prepare the initial minimum indices array for the segment tree.
        # Each element `min_indices[i]` stores the minimum index of an available basket 
        # with capacity `unique_capacities[i]`.
        min_indices = [float('inf')] * m
        for i in range(m):
            cap = unique_capacities[i]
            # Peek the minimum index from the heap if it's not empty.
            if capacity_map[cap]:
                min_indices[i] = capacity_map[cap][0] 
                
        # Step 5: Initialize and build the Segment Tree.
        # The segment tree operates on the `min_indices` array, allowing efficient querying
        # for the minimum available basket index over ranges of capacities.
        st = SegmentTree(m)
        st.build_from_array(min_indices) 

        # Step 6: Process each fruit type one by one.
        for f in fruits:
            # Find the index `k` in `unique_capacities` such that `unique_capacities[k]` is the smallest capacity >= `f`.
            # `bisect_left` finds this insertion point efficiently.
            k = bisect.bisect_left(unique_capacities, f)
            
            # If `k == m`, it means no basket has capacity >= `f`.
            if k == m: 
                unplaced_count += 1
                continue # This fruit type cannot be placed.
                
            # Query the Segment Tree for the minimum index among baskets with capacity >= `f`.
            # This corresponds to querying the range [k, m-1] in the segment tree (indices of unique capacities).
            min_idx = st.query_range(k, m - 1)
            
            # If the query returns infinity, it means no available basket in the required capacity range was found.
            if min_idx == float('inf'): 
                unplaced_count += 1 # This fruit type cannot be placed.
            else:
                # A suitable basket with index `min_idx` was found.
                # Determine the capacity of this chosen basket.
                cap = baskets[min_idx] 
                
                # Remove this basket index `min_idx` from the heap associated with its capacity `cap`.
                # This marks the basket as used. `heapq.heappop` removes the smallest element, which is `min_idx`.
                heapq.heappop(capacity_map[cap]) 
                
                # Find the new minimum index for this capacity `cap`, if any baskets of this capacity remain.
                new_min_idx = float('inf')
                if capacity_map[cap]:
                    # Peek at the top of the heap (new minimum index) without removing it.
                    new_min_idx = capacity_map[cap][0] 
                    
                # Update the Segment Tree at the leaf corresponding to this capacity `cap`.
                # First, find the index `leaf_idx` in the segment tree structure (0 to m-1).
                leaf_idx = capacity_to_idx[cap]
                # Then, update the value at this leaf index in the segment tree.
                st.update_value(leaf_idx, new_min_idx)

        # Step 7: Return the final count of unplaced fruit types.
        return unplaced_count