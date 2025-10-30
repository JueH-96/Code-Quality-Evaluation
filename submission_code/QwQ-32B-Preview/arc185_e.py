def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353
    MAX_A = 10**5
    divisors = [[] for _ in range(MAX_A + 1)]
    for i in range(1, MAX_A + 1):
        for j in range(i, MAX_A + 1, i):
            divisors[j].append(i)
    
    sum_d = {}
    dp = [0] * (N + 1)
    pow2 = [1] * N
    for i in range(1, N):
        pow2[i] = (pow2[i - 1] * 2) % MOD
    
    for m in range(1, N + 1):
        if m >= 2:
            # Compute sum over j=1 to m-1 of gcd(A_j, A_m) * 2^{j-1}
            total = 0
            for d in divisors[A[m - 1]]:
                if d in sum_d:
                    total = (total + d * sum_d[d]) % MOD
            dp[m] = (dp[m - 1] + total) % MOD
        else:
            dp[m] = 0
        # Update sum_d for A_m
        for d in divisors[A[m - 1]]:
            if d in sum_d:
                sum_d[d] = (sum_d[d] + pow2[m - 1]) % MOD
            else:
                sum_d[d] = pow2[m - 1]
    
    for m in range(1, N + 1):
        print(dp[m])

if __name__ == "__main__":
    main()