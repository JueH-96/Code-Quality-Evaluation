import sys

def solve():
    N, S = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute prefix sums P_orig[i] = sum(A[0]...A[i-1]) for i=0...N
    # P_orig[0] = 0, P_orig[i] = sum(A[0...i-1]) for i=1...N
    P_orig = [0] * (N + 1)
    for i in range(N):
        P_orig[i+1] = P_orig[i] + A[i]

    Sum_N = P_orig[N]

    # A contiguous subsequence of the infinite sequence A (A_1, A_2, ...)
    # from index l to r (1-based) has sum sum(A_l ... A_r).
    # Using 0-based indexing for the list A and infinite sequence: sum(A[l-1] ... A[r-1]).
    # This is equivalent to sum(A[(l-1)%N], A[l%N], ..., A[(r-1)%N]).
    # Let P_i = sum(A[0], A[1], ..., A[i-1]) where A[j] = A_orig[j%N]. P_0 = 0.
    # The sum from index i1 to i2 (0-based) is P_{i2+1} - P_{i1}.
    # We need to check if P_k - P_j = S for some 0 <= j < k.
    # P_i = floor(i/N) * Sum_N + P_orig[i % N] for i >= 0.

    # Consider indices k, j up to 2N. If a sum S can be formed, there exist j, k
    # such that 0 <= j < k and P_k - P_j = S.
    # P_k = q_k Sum_N + P_orig[k']
    # P_j = q_j Sum_N + P_orig[j']
    # (q_k - q_j) Sum_N + P_orig[k'] - P_orig[j'] = S, where k'=k%N, j'=j%N.
    # q = q_k - q_j. We need qN + k' > j'.
    # If we consider k, j up to 2N, then q_k, q_j are in {0, 1, 2}. q is in {-2, -1, 0, 1, 2}.
    # If q < 0: q = q_k - q_j. q_k < q_j. Minimal q_j = 1, minimal q_k=0 -> q=-1. Max q_j=2, min q_k=0 -> q=-2.
    # Case q = -1: q_k=0, q_j=1. P_k values use P_orig[0..N], P_j values use Sum_N+P_orig[0..N].
    # Need k in [0, N], j in [N+1, 2N]. k > j requires k > N+1. Max k is N. Impossible.
    # Case q = -2: q_k=0, q_j=2. P_k values use P_orig[0..N], P_j values use 2Sum_N+P_orig[0..N].
    # Need k in [0, N], j in [2N, 3N]. Max k is N. Impossible.
    # Only q >= 0 is possible.

    # If q = 0: P_orig[k'] - P_orig[j'] = S. Requires k>j and q_k=q_j. This means k = q_k N + k', j = q_k N + j'. k > j implies k' > j'.
    # So check P_orig[k'] - P_orig[j'] = S for 0 <= j' < k' <= N. (k' can be N if j' < N and k=qN+N)
    # P_orig is defined up to N. This covers subsequences entirely within A[0...N-1] or wrapping around once within the first 2N elements like A[N-1], A[N] (=A[0]).
    # Let's check P_k - P_j = S for 0 <= j < k <= 2N. This covers q_k, q_j in {0, 1, 2}. q in {0, 1, 2}.
    # Case q = 0: q_k = q_j. k>j implies k' > j'.
    # Case q = 1: q_k = q_j + 1. k>j implies N + k' > j'. Always true for N>=1.
    # Case q = 2: q_k = q_j + 2. q_j=0, q_k=2. k in [2N, 3N], j in [0, N]. k > j is possible.
    # P_k values for k up to 2N are P_0, P_1, ..., P_{2N}.
    # P_i = (i // N) * Sum_N + P_orig[i % N] for i >= 0.
    # P_0, P_1, ..., P_{2N} sequence of values:
    # P_orig[0], P_orig[1], ..., P_orig[N], Sum_N + P_orig[1], ..., Sum_N + P_orig[N], 2*Sum_N + P_orig[0]

    P_values = [0] * (2 * N + 1)
    P_values[0] = 0 # P_0

    # P_i = sum(A[0]...A[i-1] in infinite sequence)
    # P_i = floor(i/N) * Sum_N + P_orig[i % N]
    for i in range(1, 2 * N + 1):
         P_values[i] = (i // N) * Sum_N + P_orig[i % N]


    # Check if P_k - P_j = S for 0 <= j < k <= 2N
    seen_P = set()
    seen_P.add(P_values[0])

    for k in range(1, 2 * N + 1):
        if P_values[k] - S in seen_P:
            print("Yes")
            return
        seen_P.add(P_values[k])

    print("No")

solve()