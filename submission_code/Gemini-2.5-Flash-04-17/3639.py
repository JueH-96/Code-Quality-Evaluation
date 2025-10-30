from typing import List

class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)

        class SegmentTree:
            def __init__(self, arr):
                self.n = len(arr)
                # Use a size slightly larger than 4*n for safety
                tree_size = 4 * self.n
                self.tree = [0] * tree_size
                self.lazy = [0] * tree_size
                self._build(arr, 1, 0, self.n - 1)

            def _build(self, arr, v, tl, tr):
                if tl == tr:
                    self.tree[v] = arr[tl]
                else:
                    tm = (tl + tr) // 2
                    self._build(arr, 2 * v, tl, tm)
                    self._build(arr, 2 * v + 1, tm + 1, tr)
                    self.tree[v] = max(self.tree[2 * v], self.tree[2 * v + 1])

            def _push(self, v, tl, tr):
                if self.lazy[v] != 0:
                    # Apply lazy tag to tree node value
                    self.tree[v] += self.lazy[v] 
                    # Propagate lazy tag to children if not a leaf
                    if tl != tr: # Check if it's not a leaf node
                        self.lazy[2 * v] += self.lazy[v]
                        self.lazy[2 * v + 1] += self.lazy[v]
                    self.lazy[v] = 0

            def _update(self, v, tl, tr, l, r, add):
                self._push(v, tl, tr)
                if l > r or tl > r or tr < l: # Check for no overlap
                    return
                if l <= tl and tr <= r: # Complete overlap
                    self.lazy[v] += add
                    self._push(v, tl, tr)
                else: # Partial overlap
                    tm = (tl + tr) // 2
                    self._update(2 * v, tl, tm, l, r, add)
                    self._update(2 * v + 1, tm + 1, tr, l, r, add)
                    self.tree[v] = max(self.tree[2 * v], self.tree[2 * v + 1])

            def _query(self, v, tl, tr, l, r):
                if l > r or tl > r or tr < l: # Check for no overlap
                    # Return a value smaller than any possible tree value
                    return -float('inf') 
                self._push(v, tl, tr)
                if l <= tl and tr <= r: # Complete overlap
                    return self.tree[v]
                tm = (tl + tr) // 2
                return max(self._query(2 * v, tl, tm, l, r),
                           self._query(2 * v + 1, tm + 1, tr, l, r))

            def range_add(self, l, r, add):
                 if l > r: return
                 self._update(1, 0, self.n - 1, l, r, add)

            def query_max(self, l, r):
                # Adjust query range to be within the array bounds [0, n-1]
                query_l = max(0, l)
                query_r = min(self.n - 1, r)

                if query_l > query_r: # The effective query range is empty
                     return 0 # Max of an empty set of non-negative required values is 0

                # Perform the query on the segment tree
                res = self._query(1, 0, self.n - 1, query_l, query_r)
                
                # The result `res` could be negative if values have been over-decremented.
                # When we check for maximum positive required value, 0 is the floor.
                return max(0, res)


        st = SegmentTree(nums)
        
        # Process queries in reverse order
        for i in range(len(queries) - 1, -1, -1):
            l, r = queries[i]
            
            # Maximum required value in the current query range [l, r].
            # This `max_in_range` represents the minimum number of decrements
            # that must be applied to the most critical element in [l, r]
            # by queries >= i (in forward time).
            # Processing backward, we use query `i` to satisfy this requirement.
            max_in_range = st.query_max(l, r)
            
            if max_in_range > 0:
                # Decrement the entire range [l, r] by `max_in_range`.
                # This uses the current query `i` to cover the bottleneck requirements in [l, r].
                # Any required value <= max_in_range in this range is now <= 0.
                st.range_add(l, r, -max_in_range)

            # After conceptually using query `i`, any positive value outside [l, r]
            # must have been handled by queries > i (backward).
            # Check max value in [r+1, n-1] and [0, l-1]. These must be <= 0.
            # The `query_max` method handles the empty range case (returning 0).
            max_right = st.query_max(r + 1, n - 1)
            if max_right > 0:
                return False

            max_left = st.query_max(0, l - 1)
            if max_left > 0:
                 return False

        # After processing all queries backward, the maximum value in the entire array must be <= 0.
        # If any value is still positive, it means it couldn't be reduced to zero by the available queries.
        final_max = st.query_max(0, n - 1)
        return final_max <= 0