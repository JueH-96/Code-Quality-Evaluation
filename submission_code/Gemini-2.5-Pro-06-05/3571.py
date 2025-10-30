from typing import List

class Solution:
  def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
    
    # A Segment Tree for Range Maximum Query, defined locally within the method.
    class SegTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (4 * size)

        def _update(self, node, start, end, idx, val):
            if start == end:
                self.tree[node] = max(self.tree[node], val)
                return
            
            mid = (start + end) // 2
            if start <= idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

        def update(self, idx, val):
            self._update(1, 0, self.size - 1, idx, val)

        def _query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            
            if l <= start and end <= r:
                return self.tree[node]
                
            mid = (start + end) // 2
            p1 = self._query(2 * node, start, mid, l, r)
            p2 = self._query(2 * node + 1, mid + 1, end, l, r)
            
            return max(p1, p2)

        def query(self, l, r):
            if l > r:
                return 0
            return self._query(1, 0, self.size - 1, l, r)

    n = len(coordinates)
    
    # Step 1: Preprocessing
    # Coordinate compression for y-values as they can be large.
    all_y = sorted(list(set(c[1] for c in coordinates)))
    y_map = {y: i for i, y in enumerate(all_y)}
    m = len(all_y)
    
    # Attach original indices to points to update the correct dp array entry.
    points = [(c[0], c[1], i) for i, c in enumerate(coordinates)]
        
    # Step 2: Calculate dp_inc (longest path ending at each point)
    points.sort(key=lambda p: (p[0], p[1]))
    
    dp_inc = [0] * n
    tree_inc = SegTree(m)
    
    i = 0
    while i < n:
        # Group points with the same x-coordinate.
        j = i
        while j < n and points[j][0] == points[i][0]:
            j += 1
        
        # For the current group, first calculate all dp values.
        group_updates = []
        for l in range(i, j):
            _, y, original_idx = points[l]
            y_rank = y_map[y]
            max_prev_len = tree_inc.query(0, y_rank - 1)
            length = 1 + max_prev_len
            dp_inc[original_idx] = length
            group_updates.append((y_rank, length))

        # Then, update the tree with the new values from this group.
        # This ensures points with same x don't see each other as predecessors.
        for y_rank, length in group_updates:
            tree_inc.update(y_rank, length)
        
        i = j
        
    # Step 3: Calculate dp_dec (longest path starting at each point)
    points.sort(key=lambda p: (p[0], p[1]), reverse=True)
    
    dp_dec = [0] * n
    tree_dec = SegTree(m)
    
    i = 0
    while i < n:
        j = i
        while j < n and points[j][0] == points[i][0]:
            j += 1
        
        group_updates = []
        for l in range(i, j):
            _, y, original_idx = points[l]
            y_rank = y_map[y]
            max_succ_len = tree_dec.query(y_rank + 1, m - 1)
            length = 1 + max_succ_len
            dp_dec[original_idx] = length
            group_updates.append((y_rank, length))
        
        for y_rank, length in group_updates:
            tree_dec.update(y_rank, length)
            
        i = j
    
    # Step 4: Final Result
    # The length of the longest path through point k is the sum of the
    # longest path ending at k and the longest path starting at k, minus 1
    # (since k is counted in both).
    return dp_inc[k] + dp_dec[k] - 1