def solve():
    k = int(input())
    c_values = list(map(int, input().split()))
    mod = 998244353
    
    max_degree = k
    fact = [1] * (max_degree + 1)
    inv_fact = [1] * (max_degree + 1)
    
    for i in range(1, max_degree + 1):
        fact[i] = (fact[i-1] * i) % mod
        
    inv_fact[max_degree] = pow(fact[max_degree], mod - 2, mod)
    for i in range(max_degree - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i + 1)) % mod
        
    r_prev = [0] * (k + 1)
    r_prev[0] = 1
    
    for i in range(26):
        c_i = c_values[i]
        r_current = [0] * (k + 1)
        for n in range(k + 1):
            for j in range(min(n, c_i) + 1):
                term = (r_prev[n-j] * inv_fact[j]) % mod
                r_current[n] = (r_current[n] + term) % mod
        r_prev = r_current
        
    total_count = 0
    for l in range(1, k + 1):
        n_l = (fact[l] * r_prev[l]) % mod
        total_count = (total_count + n_l) % mod
        
    print(total_count)

if __name__ == '__main__':
    solve()