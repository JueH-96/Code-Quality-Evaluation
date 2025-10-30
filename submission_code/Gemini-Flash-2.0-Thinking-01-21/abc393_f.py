import sys
from bisect import bisect_left

# Segment Tree implementation
class SegmentTree:
    def __init__(self, size):
        # Tree size needs to be large enough. 4*size is a safe bet for 1-based recursive tree.
        self.size = size
        self.tree = [0] * (4 * size)

    def update(self, v, tl, tr, pos, new_val):
        # Update the max value at position 'pos' in the range [tl, tr]
        if tl == tr:
            self.tree[v] = max(self.tree[v], new_val)
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(2*v, tl, tm, pos, new_val)
            else:
                self.update(2*v+1, tm+1, tr, pos, new_val)
            # Update parent node based on children
            self.tree[v] = max(self.tree[2*v], self.tree[2*v+1])

    def query(self, v, tl, tr, l, r):
        # Query the maximum value in the range [l, r] within the current node's range [tl, tr]
        if l > r: # Empty range
            return 0
        if l == tl and r == tr:
            return self.tree[v] # Current node range is exactly the query range
        tm = (tl + tr) // 2
        # Query left and right children for the overlapping parts of the query range
        return max(self.query(2*v, tl, tm, l, min(r, tm)),
                   self.query(2*v+1, tm+1, tr, max(l, tm+1), r))

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Store queries grouped by R
    # queries_by_R[r] will store a list of (X_i, original_query_index) for queries with R_i = r
    queries_by_R = [[] for _ in range(N + 1)] # Use 1-based indexing for R
    values_to_compress = set()
    for val in A:
        values_to_compress.add(val)

    # Read queries and populate data structures
    for i in range(Q):
        R, X = map(int, sys.stdin.readline().split())
        queries_by_R[R].append((X, i))
        values_to_compress.add(X)

    # Coordinate Compression
    distinct_values = sorted(list(values_to_compress))
    M = len(distinct_values) # Number of distinct values/ranks

    # Function to get rank (1-based) of a value
    # bisect_left finds the 0-based index where the value would be inserted
    def get_rank(val):
        return bisect_left(distinct_values, val) + 1

    # Initialize Segment Tree on the ranks [1, M]
    # Segment tree operates on rank space [1, M] and stores max LIS length ending at that rank
    segment_tree = SegmentTree(M)

    # Array to store answers for each query, indexed by original query index
    ans = [0] * Q

    # Offline processing by R
    # Iterate through the array A from index 0 to N-1
    # k corresponds to the 0-based index of the element A[k]
    # The prefix considered is A[0...k], which has length k+1.
    # We process queries with R_i = k+1 after considering A[k].
    for k in range(N):
        val = A[k]
        rank_val = get_rank(val)

        # To find the maximum LIS length ending with A[k] (value val),
        # we need the maximum LIS length ending with a value strictly less than val,
        # using elements A[0..k-1]. This corresponds to querying the segment tree
        # for ranks from 1 up to rank_val - 1.
        lis_len_before = segment_tree.query(1, 1, M, 1, rank_val - 1)

        # The LIS length ending exactly with A[k] (value val) is 1 + the max length found above.
        # Update the segment tree at rank_val with this new possible length.
        # We take the maximum because we want the globally maximum length for that ending value
        # found so far across all prefixes processed up to k.
        segment_tree.update(1, 1, M, rank_val, lis_len_before + 1)

        # After processing A[k], process all queries (R_j, X_j) where R_j = k + 1.
        # The prefix considered for these queries is A[0..k].
        # The segment tree now holds LIS information considering A[0..k].
        for X_j, query_orig_idx in queries_by_R[k + 1]:
            rank_x = get_rank(X_j)
            # The answer for this query is the maximum LIS length using A[0..k]
            # where all elements are at most X_j. This is the maximum value in the
            # segment tree for ranks from 1 up to rank_x.
            ans[query_orig_idx] = segment_tree.query(1, 1, M, 1, rank_x)

    # Print results in the original query order
    for res in ans:
        sys.stdout.write(str(res) + '
')

solve()