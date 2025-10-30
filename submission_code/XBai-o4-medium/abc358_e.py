MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    K = int(input[ptr])
    ptr += 1
    C = list(map(int, input[ptr:ptr+26]))
    ptr += 26
    
    # Precompute factorial and inverse factorial up to K
    max_fact = K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Initialize DP array
    dp = [0] * (K + 1)
    dp[0] = 1
    
    for c_i in C:
        new_dp = [0] * (K + 1)
        for m_prev in range(K + 1):
            if dp[m_prev] == 0:
                continue
            max_c = min(c_i, K - m_prev)
            for c in range(0, max_c + 1):
                new_dp[m_prev + c] = (new_dp[m_prev + c] + dp[m_prev] * inv_fact[c]) % MOD
        dp = new_dp
    
    ans = 0
    for m in range(1, K + 1):
        ans = (ans + dp[m] * fact[m]) % MOD
    print(ans)

if __name__ == '__main__':
    main()