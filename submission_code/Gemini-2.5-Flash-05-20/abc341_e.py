import sys

# Setting a higher recursion limit can be useful for deeply nested recursive calls,
# but for a segment tree on 5*10^5 elements, log2(5*10^5) is about 19.
# Python's default recursion limit (usually 1000) is sufficient.
# sys.setrecursionlimit(2 * 10**5)

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()

    # Special case: If N=1, the string has only one character.
    # A single character string is always considered "good".
    # Type 1 queries (only 1 1 1 possible) don't affect an empty 'A' array.
    if N == 1:
        for _ in range(Q):
            query_parts = list(map(int, sys.stdin.readline().split()))
            query_type = query_parts[0]
            if query_type == 2:
                sys.stdout.write("Yes
")
        return

    # A[i] represents whether S_str[i] and S_str[i+1] are different.
    # A[i] = 1 if S_str[i] != S_str[i+1] (they are different)
    # A[i] = 0 if S_str[i] == S_str[i+1] (they are the same, indicating a "bad" pair)
    # The A array is 0-indexed, corresponding to S_str[0...N-1].
    # A has length N-1, covering pairs (S_str[0],S_str[1]) through (S_str[N-2],S_str[N-1]).
    A_LEN = N - 1
    A = [0] * A_LEN
    for i in range(A_LEN):
        if S_str[i] != S_str[i+1]:
            A[i] = 1
        else:
            A[i] = 0

    # Segment tree implementation
    # tree array stores the count of zeros in A for each segment.
    # Node 1 is the root of the segment tree.
    tree = [0] * (4 * A_LEN) 

    def build(node, start, end):
        """
        Builds the segment tree.
        node: current node index in the tree array.
        start, end: the range [start, end] in the A array that this node covers.
        """
        if start == end:
            # Leaf node: If A[start] is 0, it contributes 1 to the count of zeros.
            tree[node] = 1 if A[start] == 0 else 0
        else:
            mid = (start + end) // 2
            build(2 * node, start, mid)  # Build left child
            build(2 * node + 1, mid + 1, end) # Build right child
            # Internal node: sum of children's counts
            tree[node] = tree[2 * node] + tree[2 * node + 1]

    def update(node, start, end, idx):
        """
        Updates the value at index `idx` in A and propagates changes up the tree.
        node: current node index.
        start, end: range covered by this node.
        idx: index in A to be updated (0-indexed).
        """
        if start == end:
            # Leaf node for the target index: flip A[idx] and update its count.
            A[idx] = 1 - A[idx] # Flip 0 to 1, or 1 to 0
            tree[node] = 1 if A[idx] == 0 else 0 # Update count in tree based on new value
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                update(2 * node, start, mid, idx)
            else:
                update(2 * node + 1, mid + 1, end, idx)
            # After child update, recompute sum for this node.
            tree[node] = tree[2 * node] + tree[2 * node + 1]

    def query(node, start, end, l, r):
        """
        Queries the number of zeros in A within the range [l, r] (0-indexed).
        node: current node index.
        start, end: range covered by this node.
        l, r: query range [l, r] in A.
        """
        # If the current segment is completely outside the query range, return 0.
        if r < start or end < l:
            return 0
        # If the current segment is completely inside the query range, return its stored value.
        if l <= start and end <= r:
            return tree[node]
        
        # Otherwise, partially overlapping: recurse on children and sum results.
        mid = (start + end) // 2
        p1 = query(2 * node, start, mid, l, r)
        p2 = query(2 * node + 1, mid + 1, end, l, r)
        return p1 + p2

    # Initialize the segment tree with the initial A array values.
    # The segment tree operates on indices 0 to A_LEN - 1.
    build(1, 0, A_LEN - 1)

    # Process queries
    for _ in range(Q):
        query_parts = list(map(int, sys.stdin.readline().split()))
        query_type = query_parts[0]
        L, R = query_parts[1], query_parts[2] # L, R are 1-indexed from problem description

        if query_type == 1:
            # Type 1 query: Flip S[L...R] (1-indexed).
            # This operation flips S_str[L-1] through S_str[R-1] (0-indexed).
            # Based on derivation, this only affects A[L-2] and A[R-1].
            
            # Update A[L-2] if S_str[L-1] is not the first character.
            # (i.e., L-1 > 0  => L > 1).
            if L > 1:
                update(1, 0, A_LEN - 1, L - 2)
            
            # Update A[R-1] if S_str[R-1] is not the last character.
            # (i.e., R-1 < N-1 => R < N).
            if R < N:
                update(1, 0, A_LEN - 1, R - 1)

        elif query_type == 2:
            # Type 2 query: Check if S[L...R] (1-indexed) is a good string.
            # This corresponds to checking substring S_str[L-1...R-1] (0-indexed).
            
            # A string of length 1 is always good.
            if R - L + 1 == 1:
                sys.stdout.write("Yes
")
                continue

            # For a substring S_str[l_0...r_0] (0-indexed), it's good if
            # S_str[i] != S_str[i+1] for all i from l_0 to r_0-1.
            # In terms of the A array, this means A[i] must be 1 for all i in this range.
            # The relevant range for A is [L-1, R-2] (0-indexed).
            # If any A[i] in this range is 0, then the string is not good.
            # So, we query the segment tree for the number of zeros in A[L-1...R-2].
            
            # The query range [L-1, R-2] is valid (L-1 <= R-2) only if
            # L+1 <= R, which means the substring length (R-L+1) must be at least 2.
            # This is consistent with our `R-L+1 == 1` check above.
            
            num_zeros = query(1, 0, A_LEN - 1, L - 1, R - 2)
            if num_zeros == 0:
                sys.stdout.write("Yes
")
            else:
                sys.stdout.write("No
")

# Call the solve function to run the program
solve()