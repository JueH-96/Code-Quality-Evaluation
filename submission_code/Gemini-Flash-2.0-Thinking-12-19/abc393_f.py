import sys

# Increase recursion depth for deep segment tree calls
# M can be up to 2*N + Q <= 6e5. log2(6e5) is around 19. Depth is proportional to log M.
# A safe upper bound for depth is around 20-25. Set limit higher.
sys.setrecursionlimit(3000)

def coordinate_compress(values):
    """Compress unique values and return sorted unique values and mapping."""
    sorted_unique_values = sorted(list(set(values)))
    val_to_compressed_idx = {val: i + 1 for i, val in enumerate(sorted_unique_values)}
    return sorted_unique_values, val_to_compressed_idx

class SegmentTree:
    """Segment tree for range max query and point max update."""
    def __init__(self, size):
        self.size = size
        # Using 1-based indexing for tree nodes
        # Size 4*size is usually sufficient for array-based implementation
        self.tree = [0] * (4 * size + 5) # Add small buffer

    def update(self, v, tl, tr, pos, new_val):
        """Update tree at index pos with max(current_value, new_val)."""
        if tl == tr:
            self.tree[v] = max(self.tree[v], new_val)
        else:
            tm = (tl + tr) // 2
            if pos <= tm:
                self.update(2 * v, tl, tm, pos, new_val)
            else:
                self.update(2 * v + 1, tm + 1, tr, pos, new_val)
            # Parent node value is the maximum of its children
            self.tree[v] = max(self.tree[2 * v], self.tree[2 * v + 1])

    def query(self, v, tl, tr, l, r):
        """Query max in range [l, r]."""
        # If query range does not overlap with node range, return identity (0 for max)
        if r < tl or l > tr:
             return 0
        # If node range is within query range
        if l <= tl and tr <= r:
            return self.tree[v]
        # Partially overlapping ranges
        tm = (tl + tr) // 2
        return max(self.query(2 * v, tl, tm, l, r),
                   self.query(2 * v + 1, tm + 1, tr, l, r))

def solve():
    # Read N and Q
    N, Q = map(int, sys.stdin.readline().split())

    # Read array A
    A = list(map(int, sys.stdin.readline().split()))

    # Read queries and collect all values for coordinate compression
    queries = []
    all_values = set(A)
    for i in range(Q):
        R, X = map(int, sys.stdin.readline().split())
        queries.append((R, X, i))
        all_values.add(X)

    # Coordinate compression
    # Map original value to a compressed index (1-based)
    sorted_unique_values, val_to_compressed_idx = coordinate_compress(list(all_values))
    M = len(sorted_unique_values) # Number of unique values, segment tree size

    # Group queries by R
    # queries_at_R[k] stores a list of (X_i, query_index_i) for queries with R_i = k
    queries_at_R = [[] for _ in range(N + 1)]
    for R, X, query_idx in queries:
        queries_at_R[R].append((X, query_idx))

    # Initialize Segment Tree
    # Segment tree operates on compressed indices [1, M]
    # tree[j] will conceptually store max LIS length ending with value corresponding to compressed index j
    st = SegmentTree(M)

    # Array to store answers
    ans = [0] * Q

    # Process array elements A_k one by one (from k=1 to N)
    for k in range(1, N + 1):
        # Get the current element A_k (using 0-based index for A list)
        a_k = A[k - 1]
        # Get the compressed index for a_k
        p = val_to_compressed_idx[a_k]

        # Find max LIS length ending with a value strictly less than a_k
        # This corresponds to the maximum value in the segment tree for compressed indices [1, p-1]
        # The segment tree query range is [1, p-1]. If p=1, range is [1, 0], query returns 0.
        max_len_before = st.query(1, 1, M, 1, p - 1)

        # A new LIS of length max_len_before + 1 ending with a_k is possible.
        # Update the segment tree at the compressed index p with this potential new maximum length.
        st.update(1, 1, M, p, max_len_before + 1)

        # Process all queries associated with R = k
        if queries_at_R[k]:
            for x_i, query_idx in queries_at_R[k]:
                # We need the maximum length of an increasing subsequence using A[1..k]
                # where all elements in the subsequence are <= x_i.
                # An increasing subsequence ending with a value v_j <= x_i necessarily has all
                # its elements <= v_j <= x_i.
                # The maximum such length is the maximum value in the segment tree
                # for all compressed indices j such that v_j <= x_i.
                # This corresponds to querying the segment tree for max in the range [1, compressed_x_i].
                x_p = val_to_compressed_idx[x_i]
                result = st.query(1, 1, M, 1, x_p)
                ans[query_idx] = result

    # Print answers
    for result in ans:
        print(result)

# Execute the solve function
solve()