from typing import List
import bisect

# Segment Tree for Range Maximum Query and Point Max Update
class SegmentTree:
    def __init__(self, size):
        self.size = size
        # Tree nodes, initialized to 0.
        # Size 4*size is usually sufficient for a segment tree on size elements.
        self.tree = [0] * (4 * size)

    def _query(self, v, tl, tr, l, r):
        # Query range [l, r] is outside node range [tl, tr]
        if l > r or tl > r or tr < l:
            return 0
        # Node range [tl, tr] is completely inside query range [l, r]
        if l <= tl and tr <= r:
            return self.tree[v]

        # Node range overlaps with query range, recurse
        tm = (tl + tr) // 2
        left_max = self._query(2*v + 1, tl, tm, l, r)
        right_max = self._query(2*v + 2, tm + 1, tr, l, r)
        return max(left_max, right_max)

    def query(self, l, r):
        """Query maximum in compressed y-index range [l, r]"""
        # The _query method handles out-of-bounds and empty ranges.
        return self._query(0, 0, self.size - 1, l, r)

    def _update(self, v, tl, tr, pos, new_val):
        # Base case: leaf node at the position
        if tl == tr:
            self.tree[v] = max(self.tree[v], new_val)
            return

        # Recurse: find which child covers the position
        tm = (tl + tr) // 2
        if pos <= tm:
            self._update(2*v + 1, tl, tm, pos, new_val)
        else:
            self._update(2*v + 2, tm + 1, tr, pos, new_val)

        # Update current node after children updates
        self.tree[v] = max(self.tree[2*v + 1], self.tree[2*v + 2])

    def update(self, pos, new_val):
        """Update the value at compressed y-index pos with max(current, new_val)"""
        # Ensure position is within valid index range [0, size-1]
        if 0 <= pos < self.size:
            self._update(0, 0, self.size - 1, pos, new_val)


class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)

        # 1. Coordinate Compression for y-coordinates
        # Collect all y-values
        y_values = [coord[1] for coord in coordinates]
        # Get unique sorted y-values
        unique_y_values = sorted(list(set(y_values)))
        m = len(unique_y_values) # Number of unique y-coordinates

        def get_compressed_y(y):
            return bisect.bisect_left(unique_y_values, y)

        # Store points along with their original index
        points_with_idx = [(coord[0], coord[1], i) for i, coord in enumerate(coordinates)]

        # 2. Compute len_in for all points (max length ending at the point)
        # Sort points by x ascending. Points with same x will be grouped.
        sorted_for_in = sorted(points_with_idx, key=lambda p: p[0])
        len_in_values = [0] * n
        st_in = SegmentTree(m)

        i = 0
        while i < n:
            # Find group of points with the same x-coordinate
            x_coord = sorted_for_in[i][0]
            j = i
            while j < n and sorted_for_in[j][0] == x_coord:
                j += 1

            # Process points in the current group [i:j]
            # Collect updates to apply to the segment tree *after* processing the group.
            updates = []
            for idx in range(i, j):
                x, y, original_idx = sorted_for_in[idx]
                comp_y = get_compressed_y(y)

                # Query max len_in for points with x' < x and y' < y
                # Since we process groups by x, st_in only contains info from points with x' < x.
                # We need y' < y, which are compressed y-indices [0, comp_y - 1].
                max_prev_len = st_in.query(0, comp_y - 1)

                current_len_in = 1 + max_prev_len
                len_in_values[original_idx] = current_len_in

                # Store update (compressed_y, value) to apply later
                updates.append((comp_y, current_len_in))

            # Apply updates for the current group after processing all points in it.
            # This ensures points with the same x do not incorrectly extend paths to each other.
            for comp_y, val in updates:
                 st_in.update(comp_y, val)

            # Move to the next group
            i = j


        # 3. Compute len_out for all points (max length starting from the point)
        # Sort points by x descending. Points with same x will be grouped.
        sorted_for_out = sorted(points_with_idx, key=lambda p: -p[0])
        len_out_values = [0] * n
        st_out = SegmentTree(m)

        i = 0
        while i < n:
            # Find group of points with the same x-coordinate
            x_coord = sorted_for_out[i][0]
            j = i
            while j < n and sorted_for_out[j][0] == x_coord:
                j += 1

            # Process points in the current group [i:j]
            # Collect updates to apply to the segment tree *after* processing the group.
            updates = []
            for idx in range(i, j):
                x, y, original_idx = sorted_for_out[idx]
                comp_y = get_compressed_y(y)

                # Query max len_out for points with x' > x and y' > y
                # Since we process groups by x descending, st_out only contains info from points with x' > x.
                # We need y' > y, which are compressed y-indices [comp_y + 1, m - 1].
                max_next_len = st_out.query(comp_y + 1, m - 1)

                current_len_out = 1 + max_next_len
                len_out_values[original_idx] = current_len_out

                # Store update (compressed_y, value) to apply later
                updates.append((comp_y, current_len_out))

            # Apply updates for the current group after processing all points in it.
            # This ensures points with the same x do not incorrectly extend paths from each other.
            for comp_y, val in updates:
                 st_out.update(comp_y, val)

            # Move to the next group
            i = j

        # 4. Result for coordinates[k]
        # The maximum length of an increasing path containing coordinates[k] is
        # len_in[k] (max length ending at coordinates[k])
        # + len_out[k] (max length starting from coordinates[k])
        # - 1 (subtract 1 because coordinates[k] is counted in both lengths)
        return len_in_values[k] + len_out_values[k] - 1