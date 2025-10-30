import sys

# Define maximum allowed value
MAX_VAL = 10**18

# Store A and B globally
A = []
B = []
N = 0

class Node:
    def __init__(self):
        # Store list of (P, S) pairs, sorted by P
        # Represents the function max(v * P + S)
        self.lines = []

# Function to compute the upper envelope of a set of lines (P, S)
# Input lines are NOT assumed to be sorted by P initially
# Uses convex hull trick (monotonic stack)
def upper_envelope(lines):
    if not lines:
        return []

    # Sort lines by slope P. If slopes are equal, keep the one with larger S.
    lines.sort(key=lambda item: (item[0], item[1]))

    # Remove lines with same slope, keeping the one with max intercept
    distinct_lines = []
    if lines:
        distinct_lines.append(lines[0])
        for p, s in lines[1:]:
            if distinct_lines[-1][0] < p:
                distinct_lines.append((p, s))
            elif distinct_lines[-1][0] == p:
                 # Keep line with larger S if slopes are equal
                 if distinct_lines[-1][1] < s:
                     distinct_lines[-1] = (p, s)
            # Else (distinct_lines[-1][0] > p), this shouldn't happen after sorting.


    if len(distinct_lines) <= 1:
        return distinct_lines

    hull = []

    # Check function using cross product for upper hull
    # Points are (P, S). For hull[-2], hull[-1], line, pop hull[-1] if they form a right turn or are collinear.
    # Cross product of vectors (hull[-1][0] - hull[-2][0], hull[-1][1] - hull[-2][1]) and (line[0] - hull[-1][0], line[1] - hull[-1][1])
    # (p2-p1)*(s3-s2) - (p3-p2)*(s2-s1) <= 0 for right turn or collinear
    # Use arbitrary precision integers for the cross product.
    def is_non_left_turn(line1, line2, line3):
        p1, s1 = line1
        p2, s2 = line2
        p3, s3 = line3
        cross_product = (p2 - p1) * (s3 - s2) - (p3 - p2) * (s2 - s1)
        return cross_product <= 0

    for line in distinct_lines:
        # While adding 'line' makes the turn at hull[-1] non-left (right or collinear), pop hull[-1]
        while len(hull) >= 2 and is_non_left_turn(hull[-2], hull[-1], line):
            hull.pop()
        hull.append(line)

    return hull

# Merge function for segment tree
# Combines pairs L1 (for F_L(v)) and L2 (for F_R(v)) to get pairs for F_R(F_L(v))
def merge(pairs1, pairs2):
    if not pairs1:
        return upper_envelope(pairs2)
    if not pairs2:
        return upper_envelope(pairs1)

    # Generate all possible combined lines
    combined_lines = []
    for p1, s1 in pairs1:
        for p2, s2 in pairs2:
             # Compute the new pair (P, S) = (p1*p2, s1*p2 + s2)
             # Python handles arbitrary precision integers, so no explicit overflow check needed during calculation
             p = p1 * p2
             s = s1 * p2 + s2
             combined_lines.append((p, s))

    return upper_envelope(combined_lines) # Compute upper envelope of the combined lines

# Segment tree implementation
# tree[idx] stores Node for range [L, R] (inclusive, 1-based index)
tree = [None] * (4 * 100005) # Max size of tree array

def build(idx, L, R):
    global A, B
    tree[idx] = Node()
    if L == R:
        # Leaf node [L, L] represents the operation at 1-based index L
        # which uses A[L-1] and B[L-1] (0-based).
        # F'_{L, L}(v) = max(v + A[L-1], v * B[L-1])
        # The two lines are v*1 + A[L-1] and v*B[L-1] + 0
        tree[idx].lines = [(1, A[L-1]), (B[L-1], 0)]
        # Apply upper envelope to sort and prune (if needed)
        tree[idx].lines = upper_envelope(tree[idx].lines)
        return

    M = (L + R) // 2
    build(2 * idx, L, M)
    build(2 * idx + 1, M + 1, R)
    tree[idx].lines = merge(tree[2 * idx].lines, tree[2 * idx + 1].lines)

def update(idx, L, R, target_idx, new_A_val, new_B_val):
    global A, B
    if L == R:
        if target_idx == L:
             # Update global arrays (although not strictly needed by segment tree node, good practice)
             A[L-1] = new_A_val
             B[L-1] = new_B_val # This will be the old B if type 1, old A if type 2
             tree[idx].lines = [(1, new_A_val), (new_B_val, 0)]
             tree[idx].lines = upper_envelope(tree[idx].lines)
        return

    M = (L + R) // 2
    if target_idx <= M:
        update(2 * idx, L, M, target_idx, new_A_val, new_B_val)
    else:
        update(2 * idx + 1, M + 1, R, target_idx, new_A_val, new_B_val)

    tree[idx].lines = merge(tree[2 * idx].lines, tree[2 * idx + 1].lines)

# Query function
# Returns list of (P, S) pairs for F'_{query_L, query_R}(v) for 1-based indices query_L to query_R
def query(idx, L, R, query_L, query_R):
    # Query range [query_L, query_R] (1-based inclusive)
    if query_L > query_R:
        # Empty range represents identity function F(v) = v = v*1 + 0
        return [(1, 0)]

    # If current node range [L, R] is completely outside query range [query_L, query_R]
    # This case should be implicitly handled by the recursive calls not reaching here
    # if the query range is outside. Let's add an explicit check for robustness.
    if R < query_L or L > query_R:
         return [(1, 0)] # Identity function for disjoint ranges

    if query_L <= L and R <= query_R:
        # Current node range [L, R] is completely inside query range [query_L, query_R]
        return tree[idx].lines

    M = (L + R) // 2
    pairs_left = []
    pairs_right = []

    # Check if query range overlaps with left child [L, M]
    if query_L <= M:
        pairs_left = query(2 * idx, L, M, query_L, min(query_R, M))

    # Check if query range overlaps with right child [M+1, R]
    if query_R > M:
        pairs_right = query(2 * idx + 1, M + 1, R, max(query_L, M + 1), query_R)

    # Merge results from children.
    # The function for the combined range is F_right composed with F_left.
    return merge(pairs_left, pairs_right)


# Helper to evaluate the max value of v*P + S for a given v and list of (P, S) pairs
def evaluate(lines, v):
    max_val = 0 # v*P + S will be >= 0 since v >= 1, P >= 0, S >= 0

    # Evaluate v*P + S for each line and find the maximum
    for p, s in lines:
        current_val = v * p + s
        if current_val > max_val:
            max_val = current_val

    # Cap the final result at MAX_VAL = 10**18
    return min(max_val, MAX_VAL)


def main():
    global N, A, B

    # Read input using sys.stdin.readline for speed
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    Q = int(input())

    # Build segment tree over indices [1, N] (1-based)
    build(1, 1, N)

    for _ in range(Q):
        query_type, *params = map(int, input().split())

        if query_type == 1:
            i, x = params
            # Update A_i (1-based index i) to x
            A[i-1] = x # Update global array
            # Update the segment tree node corresponding to index i (ops at index i-1).
            # The node [i, i] represents F'_{i,i}(v) = max(v + A[i-1], v * B[i-1])
            update(1, 1, N, i, A[i-1], B[i-1]) # Pass current A and B values from global array
        elif query_type == 2:
            i, x = params
            # Update B_i (1-based index i) to x
            B[i-1] = x # Update global array
            # Update the segment tree node corresponding to index i (ops at index i-1).
            # The node [i, i] represents F'_{i,i}(v) = max(v + A[i-1], v * B[i-1])
            update(1, 1, N, i, A[i-1], B[i-1]) # Pass current A and B values from global array
        elif query_type == 3:
            l, r = params
            # Query type 3 l r (1-based indices)
            # Start v=0. First op at l: max(0 + A[l-1], 0 * B[l-1]) = A[l-1]
            # Then apply ops l+1, ..., r (1-based indices) starting with initial_v = A[l-1]

            initial_v = A[l-1] # 0-based A index is l-1

            # Need the function F'_{l+1, r}(v) for 1-based indices l+1 to r.
            # Query the segment tree for range [l + 1, r].
            # If l == r, the range is [r+1, r], which is empty.
            result_lines = query(1, 1, N, l + 1, r)

            # Evaluate the function represented by result_lines at initial_v
            ans = evaluate(result_lines, initial_v)
            print(ans)

# Increase recursion depth for segment tree
sys.setrecursionlimit(300000)


# Execute the main function
if __name__ == "__main__":
    main()