import bisect
from typing import List

# Segment Tree implementation for Range Maximum Query
class SegmentTree:
    """
    A Segment Tree implementation that supports point updates (setting max value)
    and range maximum queries. Initializes with -1, suitable for finding maximum
    positive sums where -1 indicates no valid sum found.
    """
    def __init__(self, size):
        """Initializes the segment tree for a range [0, size-1]."""
        # The size parameter indicates the number of leaves, corresponding to indices 0 to size-1.
        self.size = size
        # The tree array size is typically 4*size to accommodate the binary tree structure.
        self.tree = [-1] * (4 * size)

    def _update(self, v: int, tl: int, tr: int, pos: int, val: int):
        """Internal recursive function to update the tree."""
        # v: current node index (1-based)
        # tl, tr: range covered by the current node v
        # pos: the position (index) to update
        # val: the new value to consider for the update
        
        if tl == tr:
            # If we reached the leaf node corresponding to 'pos'.
            # Update the leaf node's value with the maximum seen so far.
            # Since we process points and their sums are positive, we take the max.
            self.tree[v] = max(self.tree[v], val) 
        else:
            # If not a leaf, determine whether to go left or right.
            tm = (tl + tr) // 2 # Midpoint of the current range
            if pos <= tm:
                # If position 'pos' is in the left child's range [tl, tm].
                self._update(2*v, tl, tm, pos, val) # Update left child (node 2*v)
            else:
                # If position 'pos' is in the right child's range [tm+1, tr].
                self._update(2*v+1, tm+1, tr, pos, val) # Update right child (node 2*v+1)
            
            # After updating a child, update the current node 'v'.
            # Its value is the maximum of its two children.
            self.tree[v] = max(self.tree[2*v], self.tree[2*v+1])

    def update(self, pos: int, val: int):
        """Updates the value at the given position 'pos'."""
        # Public interface for updating. Performs bounds check.
        if 0 <= pos < self.size:
             # Start update from the root node (index 1), covering the full range [0, size-1].
             self._update(1, 0, self.size - 1, pos, val)

    def _query(self, v: int, tl: int, tr: int, l: int, r: int) -> int:
        """Internal recursive function to query the maximum value in range [l, r]."""
        # v: current node index
        # tl, tr: range covered by node v
        # l, r: the query range
        
        if l > r: 
            # Base case: If the query range is invalid or does not overlap with the node's range.
            # Return -1, indicating no element found in this path.
            return -1
            
        if l == tl and r == tr:
            # Base case: If the node's range perfectly matches the query range.
            # Return the value stored in this node.
            return self.tree[v]
        
        tm = (tl + tr) // 2 # Midpoint
        # Recursively query left child for the intersection of [l, r] and [tl, tm].
        # The query range for the left child is [l, min(r, tm)].
        left_max = self._query(2*v, tl, tm, l, min(r, tm))
        
        # Recursively query right child for the intersection of [l, r] and [tm+1, tr].
        # The query range for the right child is [max(l, tm+1), r].
        right_max = self._query(2*v+1, tm+1, tr, max(l, tm+1), r)
        
        # Return the maximum of the results from the children.
        # If one path returned -1, max correctly chooses the other path's value (if positive).
        return max(left_max, right_max)

    def query(self, l: int, r: int) -> int:
        """Queries the maximum value in the range [l, r]. Handles boundary checks."""
        # Public interface for querying. Performs bounds check.
        if l >= self.size or r < 0 or l > r:
             # Check if the initial query range is fundamentally invalid.
             return -1
        
        # Clamp query range to valid indices [0, size-1]. This is defensive programming.
        l = max(0, l)
        r = min(self.size - 1, r)
        
        # After clamping, check again if the range became invalid (e.g., l > r).
        if l > r:
             return -1
             
        # Perform the query using the internal recursive function starting from the root (node 1).
        # The query range is [l, r] within the tree's total range [0, size-1].
        return self._query(1, 0, self.size - 1, l, r)


class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        """
        Finds the maximum sum nums1[j] + nums2[j] for each query [x_i, y_i]
        such that nums1[j] >= x_i and nums2[j] >= y_i. Uses offline processing
        with sorting and a Segment Tree data structure for efficiency.
        """
        n = len(nums1)
        m = len(queries)

        # Step 1: Preprocess points
        # Store points as tuples (nums1_val, nums2_val, sum_val)
        points = []
        for j in range(n):
            points.append((nums1[j], nums2[j], nums1[j] + nums2[j]))

        # Step 2: Preprocess queries
        # Store queries as tuples (x_val, y_val, original_index) to track results
        indexed_queries = []
        for i in range(m):
            indexed_queries.append((queries[i][0], queries[i][1], i))

        # Step 3: Sort points and queries for offline processing
        # Sort points by nums1 value descending. This allows adding points to the data structure
        # incrementally as we process queries sorted by x descending.
        points.sort(key=lambda x: x[0], reverse=True)

        # Sort queries by x value descending. Processing queries in this order ensures that
        # any point added satisfies the x-constraint for the current and subsequent queries.
        indexed_queries.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Coordinate compression for nums2 values
        # We need a Segment Tree indexed by nums2 values. Since nums2 values can be large,
        # we map the distinct nums2 values to indices 0..k-1.
        distinct_q_set = set(nums2) # Get unique nums2 values
        distinct_q = sorted(list(distinct_q_set)) # Sort them
        k = len(distinct_q) # Number of unique nums2 values
        
        # Check edge case: If nums2 is empty (n=0), k would be 0.
        # Constraints state n >= 1, so k is guaranteed to be at least 1.
        if k == 0: 
             # This block should not be reached under problem constraints.
             return [-1] * m

        # Create a dictionary mapping each unique nums2 value to its compressed index (0 to k-1).
        q_to_idx = {val: i for i, val in enumerate(distinct_q)}

        # Step 5: Initialize data structure
        # Initialize the Segment Tree with size k. It will maintain maximum sums for ranges of q-indices.
        st = SegmentTree(k)

        # Array to store the results for each query, initialized to -1 (default answer).
        results = [-1] * m

        # Step 6: Process queries offline using the sorted lists and Segment Tree
        point_idx = 0 # Pointer to iterate through the sorted `points` array.

        # Iterate through the queries sorted by x descending.
        for x, y, original_idx in indexed_queries:
            
            # Add relevant points to the Segment Tree.
            # These are points (p, q, s) where p >= x. Since points are sorted by p descending,
            # we iterate through `points` using `point_idx`.
            while point_idx < n and points[point_idx][0] >= x:
                # Get point details. p = nums1[j], q = nums2[j], s = p + q
                p, q, s = points[point_idx]
                
                # Find the compressed index corresponding to the q value.
                idx = q_to_idx[q]
                
                # Update the Segment Tree at this index `idx` with the sum `s`.
                # The Segment Tree's update logic ensures it stores the max sum for this q-index.
                st.update(idx, s)
                
                # Advance the point pointer. Each point is processed exactly once.
                point_idx += 1
            
            # Now query the Segment Tree for the current query (x, y).
            # We need the maximum sum 's' among points added so far where q >= y.
            # Find the smallest index `idx_y` in `distinct_q` such that `distinct_q[idx_y] >= y`.
            # `bisect_left` does this efficiently on the sorted list `distinct_q`.
            idx_y = bisect.bisect_left(distinct_q, y)

            # Check if a valid index was found. `idx_y < k` means there exists at least one distinct q >= y.
            if idx_y < k:
                # Query the Segment Tree for the maximum sum in the range [idx_y, k-1].
                # This range corresponds to all q indices where the q value is >= y.
                # The query returns the maximum sum found among eligible points (p >= x, q >= y).
                max_s = st.query(idx_y, k - 1)
                
                # Store the result for the original query index.
                results[original_idx] = max_s
            # If idx_y == k, it means no distinct q value is >= y.
            # Or, perhaps such q values exist but points with those q values haven't satisfied p >= x yet.
            # In either case, no point satisfies both conditions, so the result remains -1.
            
        # Step 7: Return results
        # The `results` array holds the answers for all queries in their original input order.
        return results