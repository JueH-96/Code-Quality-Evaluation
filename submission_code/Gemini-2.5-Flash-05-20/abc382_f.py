import sys

# Standard input/output for competitive programming
input = sys.stdin.readline

# Segment Tree with Lazy Propagation (Range Minimum Query, Range Set Update)
# Stores the minimum row (height) occupied by a bar (or the ground)
# within a column range. Initial value is H+1, representing the "ground"
# or empty space below which a bar can fall to row H.
class SegmentTree:
    def __init__(self, size, initial_val):
        self.size = size # Max column value
        # Tree nodes are 1-indexed for convenience
        self.tree = [0] * (4 * size)
        self.lazy = [0] * (4 * size)
        self.initial_val = initial_val
        self._build(1, 1, self.size, initial_val)

    # Builds the segment tree from bottom up
    def _build(self, node_idx, start, end, val):
        if start == end:
            self.tree[node_idx] = val
        else:
            mid = (start + end) // 2
            self._build(2 * node_idx, start, mid, val)
            self._build(2 * node_idx + 1, mid + 1, end, val)
            # Internal node stores the minimum of its children
            self.tree[node_idx] = min(self.tree[2 * node_idx], self.tree[2 * node_idx + 1])

    # Pushes lazy updates down to children
    def _push(self, node_idx):
        # If there's a pending lazy update for this node
        if self.lazy[node_idx] != 0:
            # Apply the update to children
            # The children's values become the lazy value
            self.tree[2 * node_idx] = self.lazy[node_idx]
            self.lazy[2 * node_idx] = self.lazy[node_idx] # Pass lazy tag to left child

            self.tree[2 * node_idx + 1] = self.lazy[node_idx]
            self.lazy[2 * node_idx + 1] = self.lazy[node_idx] # Pass lazy tag to right child

            self.lazy[node_idx] = 0 # Clear lazy tag for current node

    # Updates a range [l, r] with a new value
    def update(self, l, r, val):
        # Columns are 1-indexed in problem, so adjust range to segment tree's 1-indexed columns
        self._update_range(1, 1, self.size, l, r, val)

    def _update_range(self, node_idx, start, end, l, r, val):
        # First, apply any pending lazy update to the current node
        if self.lazy[node_idx] != 0:
            self.tree[node_idx] = self.lazy[node_idx]
            if start != end: # If not a leaf node, push to children
                self._push(node_idx)
            self.lazy[node_idx] = 0 # Clear after applying

        # If current segment is outside the update range
        if start > end or start > r or end < l:
            return

        # If current segment is fully within the update range
        if l <= start and end <= r:
            self.tree[node_idx] = val # Set its value
            if start != end: # If not a leaf node, propagate lazy tag to children
                self.lazy[2 * node_idx] = val
                self.lazy[2 * node_idx + 1] = val
            return

        # Partially overlapping, recurse on children
        mid = (start + end) // 2
        self._update_range(2 * node_idx, start, mid, l, r, val)
        self._update_range(2 * node_idx + 1, mid + 1, end, l, r, val)
        
        # After children are updated, update current node based on children's values
        self.tree[node_idx] = min(self.tree[2 * node_idx], self.tree[2 * node_idx + 1])

    # Queries for the minimum value in a range [l, r]
    def query(self, l, r):
        # Columns are 1-indexed in problem, so adjust range to segment tree's 1-indexed columns
        return self._query_range(1, 1, self.size, l, r)

    def _query_range(self, node_idx, start, end, l, r):
        # First, apply any pending lazy update to the current node
        if self.lazy[node_idx] != 0:
            self.tree[node_idx] = self.lazy[node_idx]
            if start != end: # If not a leaf node, push to children
                self._push(node_idx)
            self.lazy[node_idx] = 0 # Clear after applying

        # If current segment is outside the query range
        if start > end or start > r or end < l:
            return self.initial_val # Return initial_val (a large value like H+1) for min query

        # If current segment is fully within the query range
        if l <= start and end <= r:
            return self.tree[node_idx]

        # Partially overlapping, recurse on children
        mid = (start + end) // 2
        p1 = self._query_range(2 * node_idx, start, mid, l, r)
        p2 = self._query_range(2 * node_idx + 1, mid + 1, end, l, r)
        
        # Return the minimum of results from children
        return min(p1, p2)

def solve():
    H, W, N = map(int, input().split())

    bars_info = []
    for i in range(N):
        R, C, L = map(int, input().split())
        bars_info.append((R, C, L, i)) # Store (R, C, L, original_index)

    # Sort bars:
    # 1. By R_i in decreasing order (process bars from bottom rows upwards)
    # 2. By original_index_i in increasing order (if R_i are equal, smaller index has priority
    #    in terms of being "more fixed" when establishing final positions due to the problem's rule)
    bars_info.sort(key=lambda x: (-x[0], x[3]))

    # Initialize Segment Tree: W is max column index, H+1 is the initial value meaning "ground" below row H
    # The segment tree stores the lowest occupied row for any given column range.
    # An initial value of H+1 means that columns are empty until the ground (row H+1 being the boundary).
    segment_tree = SegmentTree(W, H + 1)

    # To store the final R' values for each bar, indexed by original bar ID
    final_R_values = [0] * N

    for R_initial, C, L, original_idx in bars_info:
        # Query the segment tree for the minimum blocking row in the bar's column range [C, C+L-1]
        # This `min_block_row` indicates the row *below* where the bar lands, or H+1 if it reaches the ground.
        min_block_row = segment_tree.query(C, C + L - 1)
        
        # The bar will land on the row just above min_block_row
        # So, its final row will be min_block_row - 1
        current_final_R = min_block_row - 1
        
        # Store the result for this bar
        final_R_values[original_idx] = current_final_R
        
        # Update the segment tree: for the columns occupied by this bar [C, C+L-1],
        # their new lowest occupied row (highest row number) is now 'current_final_R'.
        segment_tree.update(C, C + L - 1, current_final_R)

    # Print the results
    for R_prime in final_R_values:
        sys.stdout.write(str(R_prime) + "
")

# Run the solver
solve()