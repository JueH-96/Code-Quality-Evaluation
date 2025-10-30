import sys

# Increase recursion depth if needed, though not expected for this problem structure
# sys.setrecursionlimit(2000)

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # 1-based indexing for positions
    pos = [[] for _ in range(N + 1)]
    for i in range(2 * N):
        pos[A[i]].append(i + 1)

    # Identify non_adj values
    non_adj = set()
    for v in range(1, N + 1):
        # Each number appears exactly twice
        p1, p2 = pos[v]
        if abs(p1 - p2) != 1:
            non_adj.add(v)

    # Build pair_info for adjacent pairs (A_k, A_{k+1})
    # Map tuple (min(u, v), max(u, v)) to [first_k, last_k, count]
    # k is 1-based index of the first element in the pair [k, k+1]
    pair_info = {}
    for k in range(1, 2 * N): 
        u, v = A[k-1], A[k] # A is 0-indexed internally
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair not in pair_info:
            pair_info[pair] = [k, k, 1] # store [first_k, last_k, count]
        else:
            # Update last_k and count
            pair_info[pair][1] = k
            pair_info[pair][2] += 1

    # Count valid pairs (a, b)
    count = 0
    for pair, info in pair_info.items():
        a, b = pair
        k_first, k_last, m = info

        # Condition 3 check:
        # The indices k where {A_k, A_{k+1}} = {a, b} must contain at least two k's
        # such that their corresponding intervals [k, k+1] are disjoint.
        # This happens if there are at least two such k's (m >= 2) AND
        # the sequence of k's is not consecutive (k_last != k_first + m - 1).
        if m >= 2 and k_last != k_first + m - 1:
            # Conditions 1 and 2 check: a and b must be non_adj initially
            if a in non_adj and b in non_adj:
                count += 1

    print(count)

# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()