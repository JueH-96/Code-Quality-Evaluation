import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # pos[val] stores the two 0-based indices where val appears
    pos = defaultdict(list)
    for i in range(2 * N):
        pos[A[i]].append(i)

    # is_non_adjacent[val] is True if the two occurrences of val are not adjacent
    is_non_adjacent = {}
    for val in range(1, N + 1):
        # Check if the two stored indices for val are consecutive
        is_non_adjacent[val] = (pos[val][1] != pos[val][0] + 1)

    count = 0
    # Use a set to store pairs (u, v) that satisfy the index structure condition,
    # where u < v, to avoid double counting.
    # A pair {u, v} corresponds to two disjoint adjacent index blocks {i, i+1} and {j, j+1}.
    # We iterate through adjacent index pairs (i, i+1) and identify {u, v} = {A[i], A[i+1]}.
    # If the other indices of u and v, say u_other and v_other, form an adjacent pair {p, p+1},
    # then the four indices {i, i+1, p, p+1} are the positions of {u, u, v, v}.
    # This structure satisfies condition 3.
    # We only count it if both u and v are non-adjacent in the original array.
    counted_pairs = set()

    # Iterate through all adjacent index pairs (i, i+1) in the 0-indexed array A
    for i in range(2 * N - 1):
        u, v = A[i], A[i+1]

        # If A[i] == A[i+1], this value is adjacent. It cannot be part of a pair (a, b)
        # where a and b are non-adjacent in the original array. So skip.
        # Also, if u == v, {A[i], A[i+1]} is not a set of two distinct values {u, v}.
        if u == v:
            continue

        # Canonical representation for the pair {u, v} by ensuring u < v
        if u > v:
            u, v = v, u

        # If this pair {u, v} has already been processed (i.e., found from its other adjacent block), skip.
        if (u, v) in counted_pairs:
            continue

        # The current adjacent index pair is {i, i+1}. The values are {u, v}.
        # Find the other index for u and v.
        # pos[u] stores the two indices of u. One is i. The other is the sum of indices minus i.
        u_other = pos[u][0] + pos[u][1] - i
        # pos[v] stores the two indices of v. One is i+1. The other is the sum of indices minus (i+1).
        v_other = pos[v][0] + pos[v][1] - (i + 1)

        # Check if the other two indices {u_other, v_other} form an adjacent pair {p, p+1}.
        p1, p2 = min(u_other, v_other), max(u_other, v_other)

        if p2 == p1 + 1:
            # The set of four indices {i, i+1, p1, p1+1} are the positions of {u, u, v, v}.
            # This structure satisfies condition 3.
            # Now check conditions 1 and 2: u and v must be non-adjacent in the original array.
            if is_non_adjacent[u] and is_non_adjacent[v]:
                count += 1
                # Add the pair {u, v} to the set to mark it as counted.
                counted_pairs.add((u, v))

    print(count)


# Read the number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    solve()