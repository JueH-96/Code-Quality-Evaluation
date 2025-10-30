from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        q = len(queries)

        # 1. Create pairs (nums1[i], nums2[i], sum)
        points = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]

        # 2. Sort points by nums1 descending
        points.sort(key=lambda item: item[0], reverse=True)

        # 3. Store queries with original index and sort by x descending
        indexed_queries = [(queries[i][0], queries[i][1], i) for i in range(q)]
        indexed_queries.sort(key=lambda item: item[0], reverse=True)

        # 4. Coordinate compression for nums2 and query y values
        # Collect all relevant y-coordinates: all nums2[i] and all queries[i][1]
        all_y_values = sorted(list(set([p[1] for p in points] + [q[1] for q in indexed_queries])))
        coords = all_y_values
        C = len(coords)

        # 5. Build Segment Tree
        # Tree stores maximum sum in range of coordinate indices
        # Size 4*C is a safe upper bound for a segment tree on a range of size C
        tree = [-1] * (4 * C) # Initialize with -1, indicating no valid sum found yet

        def update(v, tl, tr, pos, new_val):
            """Update max value at coordinate index pos in the segment tree"""
            # v: current node index in the tree array
            # tl, tr: range of coordinate indices covered by the current node [tl, tr]
            # pos: the specific coordinate index to update
            # new_val: the new sum value to update with (take max)

            if tl == tr: # Leaf node covering the specific coordinate index
                tree[v] = max(tree[v], new_val)
            else: # Internal node
                tm = (tl + tr) // 2
                # Decide which child covers the position 'pos' and recurse
                if pos <= tm:
                    update(2 * v, tl, tm, pos, new_val) # Recurse on left child
                else:
                    update(2 * v + 1, tm + 1, tr, pos, new_val) # Recurse on right child
                # After updating children, update the current node to be the max of its children
                tree[v] = max(tree[2 * v], tree[2 * v + 1])

        def query(v, tl, tr, l, r):
            """Query max value in range [l, r] of coordinate indices in the segment tree"""
            # v: current node index
            # tl, tr: range of coordinate indices covered by the current node [tl, tr]
            # l, r: the query range of coordinate indices [l, r]

            # Base cases:
            # 1. Query range is invalid (e.g., l > r)
            # 2. Current node range [tl, tr] is completely outside the query range [l, r]
            if l > r or tl > r or tr < l:
                return -1 # Return -1 as it won't affect the max result

            # If current node range [tl, tr] is completely inside the query range [l, r]
            if l <= tl and tr <= r:
                return tree[v] # Return the precomputed max value for this node's range

            # If current node range partially overlaps with the query range, recurse
            tm = (tl + tr) // 2
            # Query the left child for the overlapping part
            left_max = query(2 * v, tl, tm, l, r)
            # Query the right child for the overlapping part
            right_max = query(2 * v + 1, tm + 1, tr, l, r)
            # The result for the current node's overlap is the max of results from children
            return max(left_max, right_max)

        # 6. Process sorted queries
        ans = [-1] * q # Initialize answer array with -1
        p_ptr = 0 # pointer for sorted points

        # Iterate through queries sorted by x descending
        for x, y, q_idx in indexed_queries:
            # Add points from the sorted 'points' list whose nums1 value >= current query's x
            # These points satisfy the first condition (nums1[j] >= x_i) and become candidates.
            # Since points are sorted by nums1 descending, we process them sequentially.
            while p_ptr < n and points[p_ptr][0] >= x:
                n1, n2, current_sum = points[p_ptr]
                # Find the coordinate index for nums2[j] using binary search on compressed coordinates
                n2_coord_idx = bisect.bisect_left(coords, n2)

                # Update the segment tree: store the max sum seen so far for this n2 coordinate.
                # The segment tree at index `n2_coord_idx` will represent points with nums2 == coords[n2_coord_idx].
                update(1, 0, C - 1, n2_coord_idx, current_sum)
                p_ptr += 1 # Move to the next point

            # For the current query (x, y), find the maximum sum among points already added (n1 >= x)
            # that also satisfy the second condition (nums2[j] >= y).
            # In the segment tree, this corresponds to querying for max sum in the range of
            # coordinate indices whose values are >= y.

            # Find the starting coordinate index for the query range [y, infinity)
            # bisect_left finds the index of the first element in coords that is >= y.
            y_coord_idx = bisect.bisect_left(coords, y)

            # Query the range [y_coord_idx, C-1] in the segment tree
            # This returns the maximum sum stored in the tree for any coordinate index >= y_coord_idx.
            # This corresponds to max sum among points (n1', n2', sum') where n1' >= x (because they were added)
            # and n2' >= y (because their coordinate index in the segment tree is >= y_coord_idx).
            res = query(1, 0, C - 1, y_coord_idx, C - 1)

            # Store the result for the original query index
            ans[q_idx] = res

        return ans