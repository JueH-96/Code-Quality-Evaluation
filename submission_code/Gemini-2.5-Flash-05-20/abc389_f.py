import sys

# Max possible initial rating and also the upper bound for L_i, R_i.
MAX_INITIAL_RATING = 5 * 10**5

# Segment Tree Node structure: (min_current_rating, max_current_rating, lazy_add)
# min_current_rating: minimum actual rating for any initial rating in this node's range
# max_current_rating: maximum actual rating for any initial rating in this node's range
# lazy_add: value to add to all current ratings in this node's range
# The size of the tree array is set to 4 * (smallest power of 2 >= MAX_INITIAL_RATING)
# For 5*10^5, the smallest power of 2 is 2^19 = 524288.
# So, array size 4 * 524288 = 2097152. Using 4 * (2**19) for clarity and safety.
tree = [(0, 0, 0)] * (4 * (2**19)) 

def build(node_idx, seg_start, seg_end):
    """
    Builds the segment tree.
    Initially, for an initial rating X, its current rating is X.
    """
    if seg_start == seg_end:
        # Leaf node: min_val and max_val are the initial rating itself, no lazy_add
        tree[node_idx] = (seg_start, seg_start, 0)
        return

    mid = (seg_start + seg_end) // 2
    build(2 * node_idx, seg_start, mid)          # Build left child
    build(2 * node_idx + 1, mid + 1, seg_end)     # Build right child

    # Internal node: min_val from left child, max_val from right child, no lazy_add initially
    tree[node_idx] = (tree[2 * node_idx][0], tree[2 * node_idx + 1][1], 0)

def push(node_idx):
    """
    Propagates lazy tag down to children.
    This should only be called for internal nodes with a non-zero lazy_add.
    """
    # If there's a lazy value to propagate
    if tree[node_idx][2] != 0:
        lazy_val = tree[node_idx][2]

        # Apply lazy_val to left child
        # Current child's (min_val, max_val) are incremented by lazy_val.
        # Current child's lazy_add is also incremented by lazy_val.
        tree[2 * node_idx] = (
            tree[2 * node_idx][0] + lazy_val,
            tree[2 * node_idx][1] + lazy_val,
            tree[2 * node_idx][2] + lazy_val
        )
        # Apply lazy_val to right child
        tree[2 * node_idx + 1] = (
            tree[2 * node_idx + 1][0] + lazy_val,
            tree[2 * node_idx + 1][1] + lazy_val,
            tree[2 * node_idx + 1][2] + lazy_val
        )
        # Reset lazy_val for current node. Its own min/max are already updated.
        tree[node_idx] = (tree[node_idx][0], tree[node_idx][1], 0)

def update(node_idx, seg_start, seg_end, contest_L, contest_R):
    """
    Updates the segment tree based on a contest (L, R).
    All initial ratings X in [seg_start, seg_end] whose current rating
    is within [contest_L, contest_R] get their current rating incremented by 1.
    """
    node_min_val, node_max_val, node_lazy = tree[node_idx]

    # Case 1: No overlap
    # The current rating range of this node is completely outside [contest_L, contest_R]
    if node_max_val < contest_L or node_min_val > contest_R:
        return

    # Case 2: Full overlap
    # The current rating range of this node is completely within [contest_L, contest_R]
    if contest_L <= node_min_val and node_max_val <= contest_R:
        # Increment min_val, max_val, and lazy_add for this node
        tree[node_idx] = (node_min_val + 1, node_max_val + 1, node_lazy + 1)
        return

    # Case 3: Partial overlap or leaf node
    # If it's a leaf node, update it directly if its current value is in range
    if seg_start == seg_end:
        if contest_L <= node_min_val <= contest_R:
            # For a leaf, min_val == max_val. Directly update it. lazy_add remains 0.
            tree[node_idx] = (node_min_val + 1, node_max_val + 1, node_lazy)
        return

    # For internal nodes with partial overlap, push lazy tag down before recursing
    push(node_idx)

    mid = (seg_start + seg_end) // 2
    update(2 * node_idx, seg_start, mid, contest_L, contest_R)          # Recurse on left child
    update(2 * node_idx + 1, mid + 1, seg_end, contest_L, contest_R)    # Recurse on right child

    # After children are updated, update min_val and max_val for current node from children.
    # The lazy_add for current node should be 0, as it was pushed down.
    left_child_min, _, _ = tree[2 * node_idx]
    _, right_child_max, _ = tree[2 * node_idx + 1]
    tree[node_idx] = (left_child_min, right_child_max, 0)


def query(node_idx, seg_start, seg_end, target_X):
    """
    Queries the segment tree for the final rating of an initial rating target_X.
    """
    # If it's a leaf node, return its current value
    if seg_start == seg_end:
        return tree[node_idx][0] # For a leaf, min_val is the current rating

    # Push lazy tag down before recursing (ensures correct values in children)
    push(node_idx)

    mid = (seg_start + seg_end) // 2
    if target_X <= mid:
        # target_X is in the left child's range
        return query(2 * node_idx, seg_start, mid, target_X)
    else:
        # target_X is in the right child's range
        return query(2 * node_idx + 1, mid + 1, seg_end, target_X)


def solve():
    # Use sys.stdin.readline for faster input
    input = sys.stdin.readline
    N = int(input())
    contests = []
    for _ in range(N):
        L, R = map(int, input().split())
        contests.append((L, R))

    # Build the segment tree with initial ratings 1 to MAX_INITIAL_RATING
    build(1, 1, MAX_INITIAL_RATING)

    # Process each contest by updating the segment tree
    for L, R in contests:
        update(1, 1, MAX_INITIAL_RATING, L, R)

    Q = int(input())
    results = []
    # Process each query
    for _ in range(Q):
        X = int(input())
        results.append(str(query(1, 1, MAX_INITIAL_RATING, X)))

    # Print all results separated by newlines
    sys.stdout.write("
".join(results) + "
")

solve()