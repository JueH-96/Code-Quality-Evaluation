# YOUR CODE HERE
import sys

# It is good practice to set a higher recursion limit for problems
# that might involve deep recursion. Although the depth here is log(N),
# which is small for the given constraints, this is a safe measure.
sys.setrecursionlimit(10**6)

# Use fast I/O
readline = sys.stdin.readline

# Read input
try:
    N_str = readline()
    if not N_str:
        # Handle empty input case
        sys.exit(0)
    N = int(N_str)
    P = list(map(int, readline().split()))
except (IOError, ValueError):
    # Handle cases with malformed input
    sys.exit(0)


# The segment tree array. 1-based indexing is used for convenience
# in calculating child indices (2*node, 2*node+1).
# A size of 4*N is sufficient for a binary tree on N leaves.
tree = [0] * (4 * N)

# The final answer array, which we will fill.
ans = [0] * N

def build(node, start, end):
    """
    Builds the segment tree. Each leaf represents an empty slot (value 1).
    An internal node stores the sum of its children (total empty slots in its range).
    """
    if start == end:
        tree[node] = 1
        return
    mid = (start + end) // 2
    build(2 * node, start, mid)
    build(2 * node + 1, mid + 1, end)
    tree[node] = tree[2 * node] + tree[2 * node + 1]

def query(node, start, end, k):
    """
    Finds the 0-based index of the k-th empty slot by traversing the tree.
    """
    if start == end:
        return start
    mid = (start + end) // 2
    
    left_sum = tree[2 * node] # Number of empty slots in the left child's range
    
    if k <= left_sum:
        # The k-th empty slot is in the left subtree.
        return query(2 * node, start, mid, k)
    else:
        # The k-th empty slot is in the right subtree.
        # We now look for the (k - left_sum)-th empty slot in that part.
        return query(2 * node + 1, mid + 1, end, k - left_sum)

def update(node, start, end, idx):
    """
    Updates the tree after a slot at index `idx` is filled.
    The value at the corresponding leaf is set to 0.
    """
    if start == end:
        tree[node] = 0
        return
    mid = (start + end) // 2
    if start <= idx <= mid:
        update(2 * node, start, mid, idx)
    else:
        update(2 * node + 1, mid + 1, end, idx)
    tree[node] = tree[2 * node] + tree[2 * node + 1]

# Initialize the segment tree representing N empty slots (indices 0 to N-1).
build(1, 0, N - 1)

# The core logic: process insertions in reverse order (from N down to 1).
# At step `i`, we place the number `i` into one of `i` available slots.
# The given position `P_i` tells us which of these `i` slots to use.
# This works because the relative order of numbers smaller than `i` is not
# affected by the insertion of `i`.
for i in range(N - 1, -1, -1):
    # The number to be placed is `i + 1`.
    # Its insertion position is `P_{i+1}`, which is `P[i]` in the 0-indexed input array.
    p_val = P[i]
    
    # Find the final 0-based index for this number using the segment tree.
    # This query finds the `p_val`-th available (empty) slot.
    idx = query(1, 0, N - 1, p_val)
    
    # Place the number `i + 1` into the determined slot.
    ans[idx] = i + 1
    
    # Mark this slot as filled by updating the segment tree.
    update(1, 0, N - 1, idx)

# Print the final array with elements separated by spaces.
print(*ans)