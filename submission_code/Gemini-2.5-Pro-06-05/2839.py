from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        q = len(queries)

        # Combine nums1 and nums2 into pairs and sort by nums1 in descending order.
        pairs = sorted([(nums1[i], nums2[i]) for i in range(n)], reverse=True)
        
        # Add original indices to queries and sort by x-constraint in descending order.
        indexed_queries = []
        for i in range(q):
            indexed_queries.append((queries[i][0], queries[i][1], i))
        indexed_queries.sort(key=lambda item: item[0], reverse=True)

        # Coordinate compression for y-values.
        all_y_coords = sorted(list(set(nums2)))
        m = len(all_y_coords)
        y_map = {y: i for i, y in enumerate(all_y_coords)}

        # Segment tree for Range Maximum Query.
        # Tree is 0-indexed: children of node i are 2i+1 and 2i+2.
        tree = [-1] * (4 * m)

        def update(node, start, end, idx, val):
            if start == end:
                tree[node] = max(tree[node], val)
                return
            
            mid = (start + end) // 2
            if start <= idx <= mid:
                update(2 * node + 1, start, mid, idx, val)
            else:
                update(2 * node + 2, mid + 1, end, idx, val)
            tree[node] = max(tree[2 * node + 1], tree[2 * node + 2])

        def query(node, start, end, l, r):
            if r < start or end < l:
                return -1
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            p1 = query(2 * node + 1, start, mid, l, r)
            p2 = query(2 * node + 2, mid + 1, end, l, r)
            return max(p1, p2)

        answers = [-1] * q
        pair_idx = 0
        
        # Process queries in their sorted order.
        for x_q, y_q, original_index in indexed_queries:
            # Add all pairs (px, py) with px >= x_q to the segment tree.
            while pair_idx < n and pairs[pair_idx][0] >= x_q:
                px, py = pairs[pair_idx]
                s = px + py
                y_idx = y_map[py]
                update(0, 0, m - 1, y_idx, s)
                pair_idx += 1
            
            # Find the starting index for y_q in the compressed y-space.
            y_query_idx = bisect.bisect_left(all_y_coords, y_q)
            
            # Query for max sum in the valid y-range.
            if y_query_idx < m:
                max_s = query(0, 0, m - 1, y_query_idx, m - 1)
                answers[original_index] = max_s
    
        return answers