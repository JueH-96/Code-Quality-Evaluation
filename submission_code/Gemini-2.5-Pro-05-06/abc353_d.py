import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # P[k] stores 10^(number of digits of A[k]) mod MOD
    # A[k] is A_k from input, using 0-based indexing for Python list A
    P = [0] * N
    for k in range(N):
        # Number of digits of A[k]
        # A[k] is positive, so A[k] >= 1. str(A[k]) is safe.
        num_digits_Ak = len(str(A[k]))
        P[k] = pow(10, num_digits_Ak, MOD)

    # SuffP[k] = sum(P[j] for j from k to N-1) mod MOD
    # SuffP will have N+1 elements. SuffP[N] is base case (sum from N to N-1 is empty, so 0).
    SuffP = [0] * (N + 1) 
    # Iterate k from N-1 down to 0
    for k in range(N - 1, -1, -1):
        SuffP[k] = (P[k] + SuffP[k+1]) % MOD
    
    # S1 = sum_{0 <= i < j <= N-1} (A[i] * 10^(digits of A[j]))
    # S1 = sum_{i=0}^{N-2} (A[i] * sum_{j=i+1}^{N-1} P[j])
    # The inner sum (sum_{j=i+1}^{N-1} P[j]) is SuffP[i+1].
    S1 = 0
    # i ranges from 0 to N-2. If N=2, i is 0. If N=1, loop doesn't run (problem says N>=2)
    for i in range(N - 1): 
        # A[i] can be up to 10^9. SuffP[i+1] is < MOD.
        # Product A[i] * SuffP[i+1] can be ~10^9 * MOD approx 10^18.
        # Python handles large integers automatically. Then take modulo.
        term = (A[i] * SuffP[i+1]) % MOD
        S1 = (S1 + term) % MOD

    # S2 = sum_{0 <= i < j <= N-1} A[j]
    # For a fixed A[j] (0-indexed), i can be 0, 1, ..., j-1. So A[j] is added j times.
    # S2 = sum_{j=1}^{N-1} (A[j] * j)
    # Note: A[0] is never added in this part of sum, as j starts from 1.
    S2 = 0
    # j ranges from 1 to N-1. If N=2, j is 1. If N=1, loop doesn't run.
    for j in range(1, N): 
        # A[j] * j can be ~10^9 * 2*10^5 = 2*10^14. Python handles.
        term = (A[j] * j) % MOD
        S2 = (S2 + term) % MOD
        
    ans = (S1 + S2) % MOD
    sys.stdout.write(str(ans) + "
")

solve()