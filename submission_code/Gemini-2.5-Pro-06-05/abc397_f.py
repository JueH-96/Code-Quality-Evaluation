# YOUR CODE HERE
import sys

# It is good practice to use fast I/O for problems with large input.
# Set a higher recursion limit for safety in case of deep recursive calls,
# though Python's default is usually sufficient for a segment tree on this scale.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * 10**5))


class SegTree:
    """
    Segment Tree with Lazy Propagation.
    Supports Range Add and Range Max Query operations.
    Operates on 0-indexed arrays.
    """
    def __init__(self, n):
        self.n = n
        # Initialize tree with 0s. In this problem, all counts are non-negative.
        self.tree = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _push(self, v):
        if self.lazy[v] != 0:
            # Apply lazy tag to children nodes
            # Left child
            self.tree[2*v] += self.lazy[v]
            self.lazy[2*v] += self.lazy[v]
            # Right child
            self.tree[2*v+1] += self.lazy[v]
            self.lazy[2*v+1] += self.lazy[v]
            self.lazy[v] = 0

    def _range_add_util(self, v, tl, tr, l, r, addval):
        if l > r:
            return
        if l == tl and r == tr:
            self.tree[v] += addval
            self.lazy[v] += addval
        else:
            self._push(v)
            tm = (tl + tr) // 2
            self._range_add_util(v*2, tl, tm, l, min(r, tm), addval)
            self._range_add_util(v*2+1, tm+1, tr, max(l, tm+1), r, addval)
            self.tree[v] = max(self.tree[v*2], self.tree[v*2+1])

    def range_add(self, l, r, addval):
        self._range_add_util(1, 0, self.n - 1, l, r, addval)

    def _range_max_util(self, v, tl, tr, l, r):
        if l > r:
            # Return a value that won't affect the maximum
            return -float('inf') 
        if l == tl and r == tr:
            return self.tree[v]
        self._push(v)
        tm = (tl + tr) // 2
        left_max = self._range_max_util(v*2, tl, tm, l, min(r, tm))
        right_max = self._range_max_util(v*2+1, tm+1, tr, max(l, tm+1), r)
        return max(left_max, right_max)

    def range_max(self, l, r):
        return self._range_max_util(1, 0, self.n - 1, l, r)
    
    def point_set(self, pos, val):
        # Effectively sets the value at a point by adding the difference.
        current_val = self.range_max(pos, pos)
        if current_val == -float('inf'):
            # If a node hasn't been touched, its value is the initial value (0).
            current_val = 0
        self.range_add(pos, pos, val - current_val)


def solve():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        N_str = input()
        if not N_str.strip(): return
        N = int(N_str)
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        return

    # --- Precomputation ---

    # L[i]: distinct count in A[0...i]
    L = [0] * N
    seen = set()
    for i in range(N):
        seen.add(A[i])
        L[i] = len(seen)

    # R[i]: distinct count in A[i...N-1]
    R = [0] * (N + 1)
    seen = set()
    for i in range(N - 1, -1, -1):
        seen.add(A[i])
        R[i] = len(seen)

    # prev[i]: 0-indexed of previous occurrence of A[i], -1 if first
    prev = [-1] * N
    last_pos = [-1] * (N + 1) # Values are 1-based, 1 to N
    for i in range(N):
        val = A[i]
        if last_pos[val] != -1:
            prev[i] = last_pos[val]
        last_pos[val] = i

    # --- Main Algorithm using Segment Tree ---
    # The segment tree operates on indices i from 0 to N-2
    seg_tree_size = N - 1
    if seg_tree_size <= 0: # This case won't happen based on N>=3
        print(0)
        return

    st = SegTree(seg_tree_size)
    max_total_sum = 0
    
    # j is the 0-indexed second split point
    # Subarrays: A[0...i], A[i+1...j], A[j+1...N-1]
    # i is from 0 to j-1. j loops from 1 to N-2
    for j in range(1, N - 1):
        # Update values based on A[j]
        pr = prev[j]
        
        # Increment scores for first split points i where i >= prev[j].
        # The range of existing i's is [0, j-2].
        update_start_i = max(0, pr)
        update_end_i = j - 2
        if update_start_i <= update_end_i:
            st.range_add(update_start_i, update_end_i, 1)

        # Set the value for the new potential first split point i = j-1
        st.point_set(j - 1, L[j - 1] + 1)
        
        # Query for max(L[i] + C(i+1, j)) for i in 0...j-1
        max_left_mid = st.range_max(0, j - 1)
        
        # Combine with the right part's distinct count
        current_total_sum = max_left_mid + R[j + 1]
        if current_total_sum > max_total_sum:
            max_total_sum = current_total_sum
            
    print(max_total_sum)

solve()