MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:27]))
    
    # Precompute factorials and inverse factorials modulo MOD
    max_fact = K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    # Initialize DP
    dp = [0] * (K + 1)
    dp[0] = 1  # Base case: empty string
    
    for ci in C:
        # Create a temporary array to hold new DP values
        tmp = dp.copy()
        for j in range(K + 1):
            if tmp[j] == 0:
                continue
            max_c = min(ci, K - j)
            for c in range(1, max_c + 1):
                if j + c > K:
                    break
                tmp[j + c] = (tmp[j + c] + dp[j] * inv_fact[c]) % MOD
        dp = tmp
    
    # Calculate the answer
    ans = 0
    for L in range(1, K + 1):
        ans = (ans + dp[L] * fact[L]) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()