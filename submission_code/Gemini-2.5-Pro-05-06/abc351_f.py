import sys
from bisect import bisect_left

# Helper class for Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        # 1-indexed tree. tree[0] is unused.
        # 'size' is the maximum 1-based index the tree will support.
        # If M unique values, ranks are 0 to M-1.
        # 1-based BIT indices corresponding to these ranks are 1 to M.
        # So, tree length is M+1.
        self.tree = [0] * (size + 1)

    def update(self, idx, val):
        # idx is 1-based (rank + 1)
        # Adds val to element at conceptual index corresponding to 'idx'
        while idx < len(self.tree):
            self.tree[idx] += val
            idx += idx & (-idx) # Move to next element to update

    def query(self, idx):
        # idx is 1-based
        # Computes sum of elements from original rank 0 up to rank (idx-1)
        # e.g. query(p) sums values for BIT indices 1...p, which corresponds to U[0]...U[p-1]
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx) # Move to previous relevant sum
        return s

# Read N from stdin
N = int(sys.stdin.readline())
# Read sequence A from stdin
A = list(map(int, sys.stdin.readline().split()))

# Coordinate compression
# U contains unique sorted values from A
U = sorted(list(set(A)))
M = len(U) # Number of unique values. Ranks will be 0 to M-1.

# Initialize Fenwick trees. They need to support 1-based indices up to M.
bit_count = FenwickTree(M)
bit_sum = FenwickTree(M)

total_sum_expr = 0

# Iterate through each element of A. Current element val_Aj plays the role of A_j.
# Previously processed elements play the role of A_i (where i < j).
for val_Aj in A:
    # p_rank_of_val_Aj is the 0-indexed rank of val_Aj in U.
    # So, val_Aj = U[p_rank_of_val_Aj].
    p_rank_of_val_Aj = bisect_left(U, val_Aj)
    
    # Query for elements strictly less than val_Aj.
    # These are U[0], ..., U[p_rank_of_val_Aj - 1].
    # The BIT query bit_*.query(p_rank_of_val_Aj) sums for 1-based indices
    # 1, ..., p_rank_of_val_Aj. These correspond to 0-indexed ranks
    # 0, ..., p_rank_of_val_Aj - 1.
    # c = count of A_i processed so far such that A_i < val_Aj
    # s = sum of A_i processed so far such that A_i < val_Aj
    c = bit_count.query(p_rank_of_val_Aj) 
    s = bit_sum.query(p_rank_of_val_Aj)
    
    # Add contribution from current val_Aj: sum of (val_Aj - A_i) for A_i < val_Aj
    # This is val_Aj * c - s
    total_sum_expr += val_Aj * c - s
    
    # Add val_Aj to Fenwick trees.
    # Its 0-indexed rank is p_rank_of_val_Aj.
    # The corresponding 1-based BIT index is p_rank_of_val_Aj + 1.
    update_bit_idx = p_rank_of_val_Aj + 1
    
    bit_count.update(update_bit_idx, 1)
    bit_sum.update(update_bit_idx, val_Aj)

print(total_sum_expr)