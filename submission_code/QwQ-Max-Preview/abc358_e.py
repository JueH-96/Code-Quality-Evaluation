MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:27]))
    
    max_fact = K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    dp = [0] * (K + 1)
    dp[0] = 1
    
    for c in C:
        new_dp = [0] * (K + 1)
        max_m = min(c, K)
        for n in range(K + 1):
            if dp[n] == 0:
                continue
            for m in range(0, min(max_m, K - n) + 1):
                inv = inv_fact[m]
                new_dp[n + m] = (new_dp[n + m] + dp[n] * inv) % MOD
        dp = new_dp
    
    ans = 0
    for n in range(1, K + 1):
        ans = (ans + dp[n] * fact[n]) % MOD
    print(ans)

if __name__ == "__main__":
    main()