# YOUR CODE HERE
import sys
input = sys.stdin.read
sys.setrecursionlimit(1000000)

def modinv(a, m):
    b, u, v = m, 1, 0
    while b:
        t = a // b
        a -= t * b
        a, b = b, a
        u -= t * v
        u, v = v, u
    u %= m
    return u

def solve(N, X, T):
    mod = 998244353
    dp = [0] * (X + 1)
    dp[0] = 1
    for i in range(1, X + 1):
        for j in range(N):
            if i >= T[j]:
                dp[i] = (dp[i] + dp[i - T[j]]) % mod
    total = sum(dp) % mod
    inv_total = modinv(total, mod)
    result = (dp[X] * modinv(N, mod) * inv_total) % mod
    return result

def main():
    data = list(map(int, input().split()))
    N = data[0]
    X = data[1]
    T = data[2:]
    print(solve(N, X, T))

if __name__ == "__main__":
    main()