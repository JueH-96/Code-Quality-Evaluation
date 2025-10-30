import sys

def read_input():
    return sys.stdin.readline().strip()

def read_ints():
    return list(map(int, read_input().split()))

def solve():
    N, M = read_ints()
    MOD = 998244353

    # Base cases
    if N == 2:
        return (M * (M - 1)) % MOD

    # Initialize variables
    a = [0] * (N + 1)
    b = [0] * (N + 1)
    a[1] = M
    b[1] = M - 1

    # Calculate a and b for each N
    for i in range(2, N + 1):
        a[i] = (a[i - 1] * (M - 1)) % MOD
        b[i] = (b[i - 1] * (M - 1)) % MOD + a[i - 1]
        b[i] %= MOD

    # Calculate the final answer
    ans = (a[N] + b[N]) % MOD
    if N % 2 == 0:
        ans = (ans - a[N // 2] + MOD) % MOD

    return ans

print(solve())