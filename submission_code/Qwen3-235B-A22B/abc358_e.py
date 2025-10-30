import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    K = int(input[0])
    C = list(map(int, input[1:27]))
    
    # Precompute factorial and inverse factorial modulo MOD up to K
    max_fact = K
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    # Initialize dp array
    dp = [0] * (K + 1)
    dp[0] = 1
    
    # Process each character's count constraint
    for c in C:
        m = min(c, K)
        new_dp = [0] * (K + 1)
        # Update new_dp using current dp and current character's generating function
        for j in range(K + 1):
            if dp[j] == 0:
                continue
            # Iterate over possible k for current character
            for k in range(m + 1):
                if j + k > K:
                    break
                new_dp[j + k] = (new_dp[j + k] + dp[j] * inv_fact[k]) % MOD
        dp = new_dp
    
    # Calculate the final answer
    ans = 0
    for L in range(1, K + 1):
        ans = (ans + dp[L] * fact[L]) % MOD
    print(ans)

if __name__ == '__main__':
    main()