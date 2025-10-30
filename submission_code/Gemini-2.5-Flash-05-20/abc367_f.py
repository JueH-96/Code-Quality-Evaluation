import sys
import random

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Generate unique random 64-bit integers for each possible value (1 to N).
    # Using a fixed seed (e.g., random.seed(42)) can make results reproducible for local testing,
    # but for competitive programming submissions, it's often omitted to use true randomness.
    
    # random_values[v] will store the unique random hash for integer value 'v'.
    # Since A_i, B_i are 1-indexed (1 to N), we need N+1 size for 1-based indexing.
    random_values = [0] * (N + 1) 
    for i in range(1, N + 1):
        # random.getrandbits(64) generates a 64-bit non-negative integer.
        random_values[i] = random.getrandbits(64) 

    # Compute prefix XOR sums for array A.
    # pxa[k] stores the XOR sum of random_values[A[0]], ..., random_values[A[k-1]].
    # pxa[0] is 0 (XOR sum of an empty prefix).
    pxa = [0] * (N + 1)
    for i in range(N):
        # A_i in problem statement is A[i] in 0-indexed list.
        # Value A[i] needs its random_value.
        pxa[i+1] = pxa[i] ^ random_values[A[i]]

    # Compute prefix XOR sums for array B, similarly.
    pxb = [0] * (N + 1)
    for i in range(N):
        pxb[i+1] = pxb[i] ^ random_values[B[i]]

    results = []
    for _ in range(Q):
        # Read query parameters (l, r, L, R are 1-based).
        l, r, L, R = map(int, sys.stdin.readline().split())

        # Calculate lengths of the subsequences.
        len_A_sub = r - l + 1
        len_B_sub = R - L + 1

        # Condition 1: Multisets must have the same number of elements.
        if len_A_sub != len_B_sub:
            results.append("No")
            continue

        # Condition 2: Multiset hashes (XOR sums of random values) must be equal.
        # The XOR sum for A[l-1 ... r-1] (inclusive, 0-indexed) is pxa[r] ^ pxa[l-1].
        hash_A = pxa[r] ^ pxa[l-1]

        # The XOR sum for B[L-1 ... R-1] (inclusive, 0-indexed) is pxb[R] ^ pxb[L-1].
        hash_B = pxb[R] ^ pxb[L-1]

        if hash_A == hash_B:
            results.append("Yes")
        else:
            results.append("No")

    # Print all results at once, separated by newlines, for faster output.
    sys.stdout.write("
".join(results) + "
")

# Call the main solve function.
solve()