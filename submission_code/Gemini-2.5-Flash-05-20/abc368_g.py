import sys

# Set recursion limit for deep segment tree calls
sys.setrecursionlimit(200000)

# Read N
N = int(sys.stdin.readline())

# Read A and B, adjusting to 0-indexed internally
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Read Q
Q = int(sys.stdin.readline())

# Define a large enough cap for intermediate values to avoid excessive growth,
# but also to signify "effectively infinity" within the problem's 10^18 constraint.
# 10^18 is the maximum allowed answer. Using 2 * 10^18 as a cap for internal P and S values.
INF_CAP = 2 * 10**18

# Segment Tree array. Each node stores a tuple (P, S):
# P: product of B_i values in the segment's range
# S: sum of A_i values in the segment's range
# Initialized with (1, 0) for identity (product 1, sum 0)
tree = [(1, 0)] * (4 * N) 

# Build the segment tree
def build(node_idx, start_idx, end_idx):
    if start_idx == end_idx:
        # Leaf node: (B[i], A[i])
        tree[node_idx] = (B[start_idx], A[start_idx])
    else:
        mid_idx = (start_idx + end_idx) // 2
        # Build left child
        build(2 * node_idx, start_idx, mid_idx)
        # Build right child
        build(2 * node_idx + 1, mid_idx + 1, end_idx)
        
        # Merge operation for (P, S) from children
        left_P, left_S = tree[2 * node_idx]
        right_P, right_S = tree[2 * node_idx + 1]
        
        # New P: product of left P and right P, capped at INF_CAP
        new_P = min(INF_CAP, left_P * right_P)
        # New S: sum of left S and right S, capped at INF_CAP
        new_S = min(INF_CAP, left_S + right_S)
        
        tree[node_idx] = (new_P, new_S)

# Update a value in the segment tree
def update(node_idx, start_idx, end_idx, target_idx, val_A, val_B):
    if start_idx == end_idx:
        # Leaf node: update A[target_idx] and B[target_idx]
        A[target_idx] = val_A
        B[target_idx] = val_B
        tree[node_idx] = (B[target_idx], A[target_idx])
    else:
        mid_idx = (start_idx + end_idx) // 2
        if start_idx <= target_idx <= mid_idx:
            # Target index is in the left child's range
            update(2 * node_idx, start_idx, mid_idx, target_idx, val_A, val_B)
        else:
            # Target index is in the right child's range
            update(2 * node_idx + 1, mid_idx + 1, end_idx, target_idx, val_A, val_B)
        
        # Re-merge after update to propagate changes upwards
        left_P, left_S = tree[2 * node_idx]
        right_P, right_S = tree[2 * node_idx + 1]
        
        new_P = min(INF_CAP, left_P * right_P)
        new_S = min(INF_CAP, left_S + right_S)
        
        tree[node_idx] = (new_P, new_S)

# Query the segment tree for a range [query_l, query_r]
# current_val is the accumulated value passed down from left to right segments
def query(node_idx, start_idx, end_idx, query_l, query_r, current_val):
    # If the current segment [start_idx, end_idx] is completely outside the query range [query_l, query_r]
    if query_r < start_idx or end_idx < query_l:
        return current_val
    
    # If the current segment [start_idx, end_idx] is completely within the query range [query_l, query_r]
    if query_l <= start_idx and end_idx <= query_r:
        # Apply the transformation of this segment node to current_val
        P_node, S_node = tree[node_idx]
        
        # Option 1: Add S_node (sum of A_k in this segment)
        val_if_add = current_val + S_node
        # Option 2: Multiply by P_node (product of B_k in this segment)
        val_if_mul = current_val * P_node
        
        # Choose the maximum of the two options. Cap result at INF_CAP.
        return min(INF_CAP, max(val_if_add, val_if_mul))
    
    # If the current segment partially overlaps, recurse on children
    mid_idx = (start_idx + end_idx) // 2
    
    # Process left child first, then pass its result to the right child,
    # ensuring operations are applied in the correct sequential order (left to right)
    val_after_left_half = query(2 * node_idx, start_idx, mid_idx, query_l, query_r, current_val)
    final_val = query(2 * node_idx + 1, mid_idx + 1, end_idx, query_l, query_r, val_after_left_half)
    
    return final_val

# Initial build of the segment tree
build(1, 0, N - 1)

# Process queries
for _ in range(Q):
    line = list(map(int, sys.stdin.readline().split()))
    query_type = line[0]

    if query_type == 1:
        # Type 1: 1 i x (Replace A_i with x)
        i, x = line[1], line[2]
        # Adjust to 0-indexed: i-1
        update(1, 0, N - 1, i - 1, x, B[i - 1])
    elif query_type == 2:
        # Type 2: 2 i x (Replace B_i with x)
        i, x = line[1], line[2]
        # Adjust to 0-indexed: i-1
        update(1, 0, N - 1, i - 1, A[i - 1], x)
    else: # query_type == 3
        # Type 3: 3 l r (Solve and print max v)
        l, r = line[1], line[2]
        # Initial value v = 0. Adjust to 0-indexed: l-1, r-1
        result = query(1, 0, N - 1, l - 1, r - 1, 0)
        # The problem guarantees answers are at most 10^18. Cap final output.
        sys.stdout.write(str(min(result, 10**18)) + '
')