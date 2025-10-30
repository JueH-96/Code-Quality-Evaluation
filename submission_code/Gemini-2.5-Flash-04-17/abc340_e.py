# YOUR CODE HERE
import sys

# Function to read space-separated integers
def read_ints():
    return list(map(int, sys.stdin.readline().split()))

# Function to read space-separated integers as long longs (Python int handles arbitrary size)
def read_long_longs():
    return list(map(int, sys.stdin.readline().split()))

# Segment tree implementation
# Size of the base array
N_seg = 0
# Tree and lazy arrays
tree = []
lazy = []

# Standard segment tree push operation
def push(v, tl, tr):
    if lazy[v] != 0:
        tm = (tl + tr) // 2
        # Update children's tree values (sum)
        tree[2*v] += lazy[v] * (tm - tl)
        tree[2v+1] += lazy[v] * (tr - tm)
        # Update children's lazy tags
        lazy[2*v] += lazy[v]
        lazy[2v+1] += lazy[v]
        # Reset parent's lazy tag
        lazy[v] = 0

# Standard segment tree range update operation [l, r)
def update_range(v, tl, tr, l, r, add_val):
    if l >= r:
        return
    if l == tl and r == tr:
        tree[v] += add_val * (tr - tl)
        lazy[v] += add_val
        return
    
    push(v, tl, tr)
    tm = (tl + tr) // 2
    update_range(2*v, tl, tm, l, min(r, tm), add_val)
    update_range(2v+1, tm, tr, max(l, tm), r, add_val)
    # Update parent's tree value based on children
    tree[v] = tree[2*v] + tree[2v+1]

# Standard segment tree point query operation (value at position pos)
def query_val(v, tl, tr, pos):
    if tl == tr - 1: # Leaf node
        return tree[v]
    
    push(v, tl, tr)
    tm = (tl + tr) // 2
    if pos < tm:
        return query_val(2*v, tl, tm, pos)
    else:
        return query_val(2v+1, tm, tr, pos)

# Helper to perform range update with wrapping [l_inc, r_inc] inclusive
def update_range_wrapped(l_inc, r_inc, add_val):
    # The range is [l_inc, r_inc + 1) in segment tree terms
    if l_inc <= r_inc:
        update_range(1, 0, N_seg, l_inc, r_inc + 1, add_val)
    else:
        # Wraps around: [l_inc, N_seg) and [0, r_inc + 1)
        update_range(1, 0, N_seg, l_inc, N_seg, add_val)
        update_range(1, 0, N_seg, 0, r_inc + 1, add_val)


# Read N and M
N, M = read_ints()

# Initialize segment tree
N_seg = N
# Allocate tree and lazy arrays. Size 4*N_seg is sufficient for 0-indexed tree.
tree = [0] * (4 * N_seg)
lazy = [0] * (4 * N_seg)

# Read initial ball counts A_i
A_initial = read_long_longs()

# Build the segment tree with initial values
# A point update at index i with value A_initial[i]
for i in range(N_seg):
    update_range(1, 0, N_seg, i, i + 1, A_initial[i])

# Read the operation boxes B_i
B_ops = read_ints()

# Perform operations
for b_op in B_ops:
    B = b_op
    
    # 1. Get current value K in box B
    K = query_val(1, 0, N_seg, B)
    
    # If K is 0, no balls to move, operation does nothing
    if K == 0:
        continue

    # 2. Take out all balls from box B (set A[B] to 0)
    # Add -K to the current value at index B
    update_range(1, 0, N_seg, B, B + 1, -K)

    # 3. Add K // N to all boxes
    k_div_n = K // N_seg
    if k_div_n > 0:
        update_range(1, 0, N_seg, 0, N_seg, k_div_n) # Add to full range [0, N_seg)

    # 4. Add 1 to boxes (B_i + C) % N for C = 1 to K % N
    rem = K % N_seg
    if rem > 0:
        l_inc = (B + 1) % N_seg # Start index inclusive
        r_inc = (B + rem) % N_seg # End index inclusive
        
        # Range to add 1 to is [l_inc, r_inc] inclusive
        update_range_wrapped(l_inc, r_inc, 1)

# Get final ball counts
final_counts = []
for i in range(N_seg):
    final_counts.append(query_val(1, 0, N_seg, i))

# Print final counts
print(*final_counts)