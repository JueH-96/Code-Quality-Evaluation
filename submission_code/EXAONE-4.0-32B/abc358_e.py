MOD = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    K = int(data[0])
    C = list(map(int, data[1:27]))
    
    max_n = K
    fact = [1] * (max_n + 1)
    inv_fact = [1] * (max_n + 1)
    
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
        
    inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
    for i in range(max_n, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % MOD

    F = [0] * (K + 1)
    F[0] = 1
    
    for c_val in C:
        max_j = min(c_val, K)
        Q_arr = [inv_fact[j] for j in range(max_j + 1)]
        
        newF = [0] * (K + 1)
        for m in range(0, K + 1):
            total = 0
            j_low = max(0, m - max_j)
            for j in range(j_low, m + 1):
                idx = m - j
                total = (total + F[j] * Q_arr[idx]) % MOD
            newF[m] = total
        F = newF

    ans = 0
    for n in range(1, K + 1):
        ans = (ans + fact[n] * F[n]) % MOD
        
    print(ans)

if __name__ == "__main__":
    main()