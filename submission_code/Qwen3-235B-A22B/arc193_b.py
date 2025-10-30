MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    s = input[1].strip()
    K = sum(1 for c in s if c == '1')
    
    # Precompute factorial and inverse factorial modulo MOD up to N
    max_n = N
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n-1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
    
    sum_seq = 0
    fact_n = fact[N]
    max_t = N // 2
    for t in range(0, max_t + 1):
        rem = N - 2 * t
        # Compute term = fact_n / ( (t! )^2 * (rem)! ) mod MOD
        term = fact_n * inv_fact[t] % MOD
        term = term * inv_fact[t] % MOD
        term = term * inv_fact[rem] % MOD
        sum_seq = (sum_seq + term) % MOD
    
    pow2 = pow(2, K, MOD)
    ans = sum_seq * pow2 % MOD
    print(ans)

if __name__ == "__main__":
    main()