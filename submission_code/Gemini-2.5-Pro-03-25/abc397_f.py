# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep segment tree calls
# Necessary for large N due to recursive nature of segment tree operations
try:
    # Set a large enough limit. Based on N<=3e5, log2(N) approx 18. Set higher for safety.
    sys.setrecursionlimit(2000000) 
except Exception as e: 
    # Optional: print error message if setting recursion limit fails
    # print(f"Could not set recursion depth: {e}", file=sys.stderr)
    pass

class SegmentTree:
    """ 
    Lazy Segment Tree supporting Range Maximum Query and Range Add Update.
    Used here to efficiently find max(L(i) + M(i, j)) for fixed j over relevant i.
    """
    def __init__(self, data):
        """ Initializes the segment tree with initial data `data`. """
        self.N = len(data)
        # Handle empty data case to avoid errors
        if self.N == 0: 
            self.tree = []
            self.lazy = []
            return
        # Allocate memory for tree and lazy arrays. 4*N is usually sufficient.
        self.tree = [0] * (4 * self.N)
        self.lazy = [0] * (4 * self.N)
        # Build the tree if data is provided
        if data: 
             self._build(data, 0, 0, self.N - 1)

    def _build(self, data, node, start, end):
        """ Recursively builds the segment tree from initial data. """
        if start == end:
            # Leaf node stores the value from the initial data array
            self.tree[node] = data[start]
        else:
            mid = (start + end) // 2
            # Recursively build left and right children
            self._build(data, 2 * node + 1, start, mid)
            self._build(data, 2 * node + 2, mid + 1, end)
            # Internal node stores the maximum of its children
            self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def _apply_lazy(self, node, start, end):
         """ Applies the lazy value at a node and propagates it to children if it's not a leaf. """
         if self.lazy[node] != 0:
            # Add lazy value to the node's current value
            self.tree[node] += self.lazy[node]
            # If it's not a leaf node, pass the lazy value down to children
            if start != end:  
                self.lazy[2 * node + 1] += self.lazy[node]
                self.lazy[2 * node + 2] += self.lazy[node]
            # Reset lazy value for the current node as it has been applied/propagated
            self.lazy[node] = 0 
    
    def _update_range(self, node, start, end, l, r, val):
        """ Internal recursive method to perform range update (add `val` to range [l, r]). """
        self._apply_lazy(node, start, end) # Apply any pending lazy updates first

        # If current segment [start, end] is completely outside the update range [l, r]
        if start > end or start > r or end < l: 
            return

        # If current segment is fully within the update range [l, r]
        if l <= start and end <= r: 
            # Add `val` to the node's value
            self.tree[node] += val
            # If it's not a leaf, mark children with lazy value `val` for future propagation
            if start != end: 
                self.lazy[2 * node + 1] += val
                self.lazy[2 * node + 2] += val
            return

        # If current segment partially overlaps with update range [l, r]
        mid = (start + end) // 2
        # Recurse on left and right children
        self._update_range(2 * node + 1, start, mid, l, r, val)
        self._update_range(2 * node + 2, mid + 1, end, l, r, val)
        
        # After updating children, update current node's value to be the max of its children's updated values
        self.tree[node] = max(self.tree[2 * node + 1], self.tree[2 * node + 2])

    def update(self, l, r, val):
         """ Public method to update range [l, r] by adding `val`. Handles index clamping. """
         if self.N == 0 or l > r: return # Check for empty tree or invalid range
         # Clamp query range [l, r] to valid tree indices [0, N-1]
         l = max(0, l)
         r = min(self.N - 1, r)
         # If range becomes invalid after clamping (l > r), do nothing
         if l > r: return 
         self._update_range(0, 0, self.N - 1, l, r, val)

    def _query_range(self, node, start, end, l, r):
        """ Internal recursive method to perform range maximum query on [l, r]. """
        # If current segment [start, end] is completely outside the query range [l, r]
        if start > end or start > r or end < l: 
            # Return negative infinity as identity for max operation
            return -float('inf') 
        
        self._apply_lazy(node, start, end) # Apply any pending lazy updates before querying

        # If current segment is fully within query range [l, r]
        if l <= start and end <= r: 
            # Return the node's current value
            return self.tree[node]

        # If current segment partially overlaps query range [l, r]
        mid = (start + end) // 2
        # Query left and right children and return the maximum result
        p1 = self._query_range(2 * node + 1, start, mid, l, r)
        p2 = self._query_range(2 * node + 2, mid + 1, end, l, r)
        return max(p1, p2)

    def query(self, l, r):
         """ Public method to query maximum value in range [l, r]. Handles index clamping. """
         if self.N == 0 or l > r: return -float('inf') # Check for empty tree or invalid range
         # Clamp query range [l, r] to valid tree indices [0, N-1]
         l = max(0, l)
         r = min(self.N - 1, r)
         # If range becomes invalid after clamping (l > r), return -inf
         if l > r: return -float('inf') 
         return self._query_range(0, 0, self.N - 1, l, r)

def solve():
    """ Reads input, performs calculations, and prints the result. """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Base case: N=3. The only possible partition is (A[0]), (A[1]), (A[2]).
    # The count of distinct elements in each subarray is 1. Total sum is always 3.
    if N == 3:
        print(3)
        return

    # Precompute L(i): count of distinct elements in A[0...i]
    # L array has size N-1, covering indices i = 0 to N-2.
    L = [0] * (N - 1)
    seen_L = set()
    for i in range(N - 1):
        seen_L.add(A[i])
        L[i] = len(seen_L)

    # Precompute R(j): count of distinct elements in A[j+1...N-1]
    # R array has size N-1, covering indices j = 0 to N-2.
    R = [0] * (N - 1)
    seen_R = set()
    # Compute R[j] values by iterating j downwards from N-2 to 0.
    for j in range(N - 2, -1, -1): 
        seen_R.add(A[j+1])
        R[j] = len(seen_R)

    # Precompute P(k): 0-based index of the previous occurrence of A[k].
    # Store -1 if A[k] is the first occurrence.
    # P array has size N, covering indices k = 0 to N-1.
    P = [-1] * N
    last_seen = {} # Dictionary to store the last seen index of each value
    for k in range(N):
        val = A[k]
        if val in last_seen:
            P[k] = last_seen[val]
        last_seen[val] = k

    # Initialize the Segment Tree with the precomputed L values.
    # The tree structure will cover indices i from 0 to N-2.
    st = SegmentTree(L)
    
    # Initialize the maximum total sum found so far.
    max_total_sum = 0

    # Iterate through possible second split points `j`.
    # `j` ranges from 1 to N-2, corresponding to splitting after index `j`.
    # The three subarrays are A[0..i], A[i+1..j], A[j+1..N-1] where 0 <= i < j.
    for j in range(1, N - 1):
        # Element A[j] is considered. If it contributes a new distinct value to M(i, j), update scores.
        p_j = P[j] # Get the index of the previous occurrence of A[j].

        # Update step: Increment scores in the segment tree.
        # A[j] adds 1 to the distinct count M(i, j) compared to M(i, j-1)
        # if and only if A[j] did not appear in A[i+1...j-1].
        # This condition is equivalent to P[j] <= i (using 0-based indexing).
        # We need to increment the values associated with indices i in the range [P[j], j-1].
        
        # Determine the start index for the update. If P[j] is -1 (first occurrence), start from 0.
        update_start_idx = 0 
        if p_j != -1:
             update_start_idx = p_j
             
        update_end_idx = j - 1 # The update range ends at j-1.
        
        # Perform the range update only if the range is valid (start <= end).
        if update_start_idx <= update_end_idx:
             st.update(update_start_idx, update_end_idx, 1)
        
        # Query step: Find the maximum value in the segment tree over the range [0, j-1].
        # This maximum value represents V_j = max_{0 <= i < j} (L(i) + M(i, j)).
        # The value L(i) was the initial value, and M(i,j) contribution has been added incrementally.
        V_j = st.query(0, j - 1)
        
        # Check if query returned a valid maximum (not -inf).
        # This check is mostly a safeguard; for N>=3 and j>=1, the query range [0, j-1] is non-empty.
        if V_j != -float('inf'):
            # Calculate the total sum for the best split ending at j: V_j + R(j).
            current_total_sum = V_j + R[j]
            # Update the overall maximum sum found.
            max_total_sum = max(max_total_sum, current_total_sum)

    # Print the final maximum sum.
    print(max_total_sum)

# Execute the main solution function.
solve()