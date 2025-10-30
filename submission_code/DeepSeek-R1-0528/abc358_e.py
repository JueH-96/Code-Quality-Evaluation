mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    K = int(data[0])
    caps = list(map(int, data[1:1+26]))
    
    max_n = K
    fact = [1] * (max_n+1)
    inv_fact = [1] * (max_n+1)
    
    for i in range(1, max_n+1):
        fact[i] = fact[i-1] * i % mod
        
    inv_fact[max_n] = pow(fact[max_n], mod-2, mod)
    for i in range(max_n, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % mod
        
    dp = [0] * (K+1)
    dp[0] = 1
    
    for cap in caps:
        new_dp = [0] * (K+1)
        for j in range(0, K+1):
            max_k = min(cap, j)
            total = 0
            for k in range(0, max_k+1):
                term = dp[j-k] * inv_fact[k]
                total = (total + term) % mod
            new_dp[j] = total % mod
        dp = new_dp
        
    ans = 0
    for j in range(1, K+1):
        term = dp[j] * fact[j] % mod
        ans = (ans + term) % mod
        
    print(ans)

if __name__ == '__main__':
    main()