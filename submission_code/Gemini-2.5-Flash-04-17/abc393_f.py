import sys
from collections import defaultdict
import bisect

# Increase recursion depth if needed (usually not needed for this problem structure)
# sys.setrecursionlimit(2000)

class FenwickTree:
    def __init__(self, size):
        # size is the number of distinct compressed coordinates (M)
        # The internal tree array uses 1-based indexing, so size + 1
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, val):
        # Update the maximum value at conceptual array index `idx` (0-based compressed coordinate)
        # `arr[idx] = max(arr[idx], val)`
        # Propagates the maximum value upwards in the tree structure
        idx += 1 # Convert to 1-based index for internal tree array
        while idx <= self.size:
            self.tree[idx] = max(self.tree[idx], val)
            idx += idx & -idx

    def query(self, idx):
        # Query the maximum value in the conceptual array range [0, idx] (0-based compressed coordinates)
        # `max(arr[0 ... idx])`
        # Propagates the maximum value downwards in the tree structure
        idx += 1 # Convert to 1-based index for internal tree array
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & -idx
        return res

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())

    # Read array A
    A = list(map(int, sys.stdin.readline().split()))

    # Read queries and store them, grouping by R
    queries = []
    all_values = set() # Set to collect all unique values for coordinate compression
    all_values.update(A) # Add values from A

    # Store queries along with their original index
    queries_at_R = defaultdict(list)
    for i in range(Q):
        R, X = map(int, sys.stdin.readline().split())
        # Store query as (X, original_index)
        # R is 1-based in input, convert to 0-based index for processing A
        queries_at_R[R - 1].append((X, i))
        all_values.add(X) # Add X values from queries

    # Coordinate Compression
    # Create sorted list of unique values
    sorted_unique_values = sorted(list(all_values))
    # Create a mapping from original value to compressed rank (0-based)
    value_to_rank = {v: rank for rank, v in enumerate(sorted_unique_values)}
    M = len(sorted_unique_values) # Number of unique unique_values

    # Initialize Fenwick tree
    # The size of the FT is M, corresponding to compressed coordinates 0 to M-1
    ft = FenwickTree(M)

    # Array to store results for each query
    results = [0] * Q

    # Process elements of A one by one
    for i in range(N):
        v = A[i]
        c_v = value_to_rank[v] # Get compressed coordinate of A[i] (0-based)

        # Find the maximum length of an increasing subsequence using A[0...i-1]
        # ending with a value strictly less than v.
        # This is the maximum value in the conceptual array for compressed coordinates 0 up to c_v - 1.
        # If c_v is 0 (v is the smallest unique value), there are no values strictly less than v among unique values,
        # so the max length is 0.
        max_len_ending_before = ft.query(c_v - 1) if c_v > 0 else 0

        # The maximum length of an increasing subsequence using A[0...i] ending with v is
        # 1 + the maximum length found above.
        current_lis_len = 1 + max_len_ending_before

        # Update the BIT. The conceptual array entry at c_v should store the maximum LIS length ending with value v
        # found so far using elements A[0...i].
        # The FenwickTree.update method implements `arr[c_v] = max(arr[c_v], current_lis_len)`,
        # ensuring we store the maximum length if v appeared before or if a longer sequence ending in v is found now.
        ft.update(c_v, current_lis_len)

        # After processing A[i], answer all queries that have R = i + 1
        if i in queries_at_R:
            for X, query_idx in queries_at_R[i]:
                c_X = value_to_rank[X] # Get compressed coordinate of X

                # The answer to the query (R=i+1, X) is the maximum length of an LIS
                # using A[0...i] where all elements in the subsequence are <= X.
                # An LIS with elements <= X must end with some value v' <= X.
                # The maximum length among all such possibilities is the maximum LIS length
                # ending with any value whose compressed coordinate is <= c_X.
                # This is exactly the result of ft.query(c_X).
                results[query_idx] = ft.query(c_X)

    # Print the results for all queries
    for res in results:
        sys.stdout.write(str(res) + '
')

# Call the solve function
solve()