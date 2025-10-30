import sys

def solve():
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split())) # A is A_1, A_2, ..., A_N

    # Based on sample cases, the number of permutations appears to be the product
    # (1 - A_1) * (2 - A_2) * ... * (N - A_N) modulo 998244353.
    # In the 0-indexed Python list A, A[i-1] corresponds to A_i.
    # So the product is (1 - A[0]) * (2 - A[1]) * ... * (N - A[N-1]).

    ans = 1
    for i in range(N):
        # i is the 0-indexed loop variable, from 0 to N-1.
        # The corresponding problem index is i+1.
        # The value A_{i+1} is A[i].
        # The term in the product is (i+1) - A_{i+1} = (i+1) - A[i].
        term = (i + 1) - A[i]
        ans = (ans * term) % MOD

    print(ans)

solve()