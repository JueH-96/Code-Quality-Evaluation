import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1

    groups = defaultdict(set)
    for _ in range(M):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1
        X = int(data[idx])
        idx += 1
        groups[(L, R)].add(X)

    # Precompute factorials modulo MOD
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i-1] * i % MOD

    total_violating = 0
    for (L, R), Xs in groups.items():
        m = len(Xs)
        k = R - L + 1
        inv_k = pow(k, MOD-2, MOD)
        term = m * fact[N] % MOD
        term = term * inv_k % MOD
        total_violating = (total_violating + term) % MOD

    ans = (fact[N] - total_violating) % MOD
    print(ans)

if __name__ == '__main__':
    main()