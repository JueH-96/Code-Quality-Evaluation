# YOUR CODE HERE
def solve():
    import sys
    MOD = 998244353
    N, M = map(int, sys.stdin.read().split())
    if N ==1:
        C = M * M
    else:
        NM = N * M
        C = N * M * (NM +1) //2
    print(C % MOD)