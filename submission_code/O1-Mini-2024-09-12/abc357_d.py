# YOUR CODE HERE
def solve():
    import sys
    M = 998244353
    N_str = sys.stdin.read().strip()
    N = int(N_str)
    D = len(N_str)
    r = pow(10, D, M)
    if r != 1:
        numerator = pow(r, N, M) - 1
        inv = pow(r - 1, M - 2, M)
        s = (numerator * inv) % M
    else:
        s = N % M
    result = (N % M) * s % M
    print(result)