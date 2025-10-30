mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    binom = [[0] * (K+1) for _ in range(K+1)]
    for k in range(0, K+1):
        binom[k][0] = 1
        if k >= 1:
            binom[k][k] = 1
            for j in range(1, k):
                binom[k][j] = (binom[k-1][j-1] + binom[k-1][j]) % mod

    dp = [0] * (K+1)
    ans = 0
    for a in A:
        pow_a = [1] * (K+1)
        if K >= 1:
            pow_a[1] = a % mod
            for exp in range(2, K+1):
                pow_a[exp] = pow_a[exp-1] * a % mod

        new_dp = [0] * (K+1)
        for k in range(0, K+1):
            term1 = pow_a[k]
            term2 = 0
            for j in range(0, k+1):
                term2 = (term2 + binom[k][j] * pow_a[k-j] % mod * dp[j]) % mod
            new_dp[k] = (term1 + term2) % mod

        ans = (ans + new_dp[K]) % mod
        dp = new_dp

    print(ans % mod)

if __name__ == "__main__":
    main()