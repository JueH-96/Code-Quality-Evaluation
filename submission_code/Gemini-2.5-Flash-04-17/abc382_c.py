import sys

# Function to build the segment tree
def build(v, tl, tr, A, tree):
    if tl == tr:
        tree[v] = A[tl]
    else:
        tm = (tl + tr) // 2
        build(2*v, tl, tm, A, tree)
        build(2*v+1, tm+1, tr, A, tree)
        tree[v] = min(tree[2*v], tree[2*v+1])

# Function to find the first index i (0-based) in A[tl:tr+1] <= threshold
# This search prioritizes the minimum index.
def find_first_optimized(v, tl, tr, threshold, tree):
    # If the minimum in the current range is greater than threshold, no element satisfies
    # This check at the start is crucial for efficiency and correctness of the search.
    if tree[v] > threshold:
        return -1

    # Base case: leaf node. We know A[tl] <= threshold. Return the index.
    if tl == tr:
        return tl

    # Internal node
    tm = (tl + tr) // 2

    # Check if the left child's range [tl, tm] contains any element <= threshold.
    # The minimum value in the left child's range is tree[2*v].
    if tree[2*v] <= threshold:
        # There is at least one element <= threshold in the left half.
        # The first such element in the overall range [tl, tr] must be in the left half
        # because indices are ordered. So, recursively search the left child.
        return find_first_optimized(2*v, tl, tm, threshold, tree)
    else:
        # No element <= threshold in the left half [tl, tm].
        # The first such element (if any) must be in the right half [tm+1, tr].
        # Recursively search the right child.
        return find_first_optimized(2*v+1, tm+1, tr, threshold, tree)

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# Read A (gourmet levels)
# Store as a list, 0-indexed internally
A = list(map(int, sys.stdin.readline().split()))

# Read B (sushi deliciousness)
B = list(map(int, sys.stdin.readline().split()))

# Segment Tree setup
# A safe upper bound for segment tree size is 4*N
tree_size = 4 * N
# Initialize tree with a value larger than any possible A_i or B_j
# float('inf') works well for minimum queries.
tree = [float('inf')] * tree_size

# Build the tree
# Constraints guarantee N >= 1, so A is not empty.
# The tree covers the 0-based indices [0, N-1].
build(1, 0, N-1, A, tree)

# Process each sushi
results = []
for deliciousness in B:
    # Query the segment tree for the first person (0-indexed) whose gourmet level
    # is less than or equal to the current sushi's deliciousness.
    # The search range is always the full person range [0, N-1].
    person_idx_0based = find_first_optimized(1, 0, N-1, deliciousness, tree)

    if person_idx_0based != -1:
        # Found a person. Convert the 0-based index to the 1-based person number.
        results.append(person_idx_0based + 1)
    else:
        # If find_first_optimized returns -1, it means no person could eat this sushi.
        results.append(-1)

# Print the results, one per line.
for result in results:
    print(result)