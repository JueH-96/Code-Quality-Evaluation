import sys
from collections import defaultdict

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Map values to a smaller range
    all_values = sorted(list(set(A + B)))
    value_map = {v: i for i, v in enumerate(all_values)}
    M = len(all_values) # Number of unique values

    # Compute prefix sums for A and B
    # pa[v_idx][i] = count of value mapped to v_idx in A[0...i-1]
    pa = [[0] * (N + 1) for _ in range(M)]
    pb = [[0] * (N + 1) for _ in range(M)]

    # Build prefix count arrays (O(N * M))
    for i in range(N):
        for v_idx in range(M):
            pa[v_idx][i+1] = pa[v_idx][i]
            pb[v_idx][i+1] = pb[v_idx][i]
        pa[value_map[A[i]]][i+1] += 1
        pb[value_map[B[i]]][i+1] += 1

    # Check conditions using 1-based indexing for logic, 0-based for array access
    # Conditions derived for 1-based array indices [1..N]

    # Prefix condition: count(B[1..i])(v) <= count(A[1..min(N, i+K)])(v)
    # Corresponds to pb[v_idx][i] <= pa[v_idx][min(N, i+K)]
    # O(M * N) check
    for v_idx in range(M):
        for i in range(1, N + 1):
            if pb[v_idx][i] > pa[v_idx][min(N, i + K)]:
                return "No"

    # Suffix condition: count(B[i..N])(v) <= count(A[max(1, i-K)..N])(v)
    # Corresponds to (pb[v_idx][N] - pb[v_idx][i-1]) <= (pa[v_idx][N] - pa[v_idx][max(1, i-K)-1])
    # O(M * N) check
    for v_idx in range(M):
        for i in range(1, N + 1):
             # Count in B from index i to N (1-based)
            count_B_suffix = pb[v_idx][N] - pb[v_idx][i-1]

            # Count in A from index max(1, i-K) to N (1-based)
            # max(1, i-K) corresponds to 0-based index max(0, i-K-1)
            start_A_suffix_0based = max(0, i - K - 1)
            count_A_suffix = pa[v_idx][N] - pa[v_idx][start_A_suffix_0based]

            if count_B_suffix > count_A_suffix:
                 return "No"

    return "Yes"


T = int(sys.stdin.readline())
results = []
for _ in range(T):
    results.append(solve())

for res in results:
    print(res)