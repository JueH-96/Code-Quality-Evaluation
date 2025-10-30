import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Calculate powers of 10 modulo MOD for each A[j]
    # p[j] will store 10^(number of digits in A[j]) mod MOD
    p = [0] * N
    for j in range(N):
        # The number of digits in a positive integer x is len(str(x)).
        p[j] = pow(10, len(str(A[j])), MOD)

    # Calculate suffix sums of p
    # s[k] will store sum(p[k]...p[N-1]) % MOD
    # This sum corresponds to sum(10^digits(A_l)) for A_l appearing after A_k
    s = [0] * (N + 1) # Use size N+1 for convenience in indexing s[i+1]
    # s[N] is conceptually 0 (sum over empty range)
    s[N] = 0 
    for k in range(N - 1, -1, -1):
        s[k] = (p[k] + s[k+1]) % MOD

    # Calculate the first term: sum(A[i] * 10^digits(A_j)) over 0 <= i < j <= N-1
    # This can be grouped by A[i]: sum(A[i] * sum(10^digits(A[j]) for j > i)) for i from 0 to N-2
    # sum(10^digits(A[j]) for j > i) is sum(p[j] for j from i+1 to N-1), which is s[i+1]
    term1 = 0
    for i in range(N - 1):
        # A[i] is the left part (A_i). It is multiplied by the sum of powers of 10
        # corresponding to elements appearing after it (A_j for j > i).
        # The indices j > i (0-based) correspond to 1-based indices from i+1 to N.
        # The sum of 10^digits(A[j]) for j from i+1 to N-1 is s[i+1].
        term1 = (term1 + (A[i] * s[i+1]) % MOD) % MOD

    # Calculate the second term: sum(A_j) over 0 <= i < j <= N-1
    # This can be grouped by A[j]: sum(A[j] for i from 0 to j-1) for j from 1 to N-1
    # For a fixed A[j] (at 0-based index j, where j >= 1), it appears as the right part
    # in pairs (A[i], A[j]) where i goes from 0 to j-1.
    # There are j such values of i. So A[j] is added j times.
    term2 = 0
    for j in range(1, N):
        # A[j] is the right part (A_j). It is added j times (for i=0, ..., j-1).
        term2 = (term2 + (j * A[j]) % MOD) % MOD

    total_sum = (term1 + term2) % MOD

    print(total_sum)

solve()