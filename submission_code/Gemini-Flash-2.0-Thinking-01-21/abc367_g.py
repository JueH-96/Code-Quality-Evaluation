def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    mod = 998244353
    
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    def inverse(n):
        return power(n, mod - 2)
        
    primitive_root = 3
    omega_base = power(primitive_root, (mod - 1) // m)
    
    total_sum = 0
    
    for j in range(m):
        omega_j = power(omega_base, j)
        h = [0] * (k + 1)
        
        h0_val = 1
        for val in a:
            h0_val = (h0_val * (1 + omega_j)) % mod
        h[0] = h0_val
        
        u1_val = 0
        for i in range(n):
            term_denominator = (1 + omega_j) % mod
            term_numerator = (omega_j * a[i]) % mod
            term = (term_numerator * inverse(term_denominator)) % mod
            u1_val = (u1_val + term) % mod
        
        u2_val = 0
        for i in range(n):
            term_denominator = power((1 + omega_j) % mod, 2)
            term_numerator = (omega_j * (a[i]**2)) % mod
            term = (term_numerator * inverse(term_denominator)) % mod
            u2_val = (u2_val + term) % mod
            
        if k >= 1:
            h[1] = (h[0] * u1_val) % mod
        if k >= 2:
            for current_k in range(1, k):
                h[current_k+1] = (h[current_k] * u1_val + h[current_k-1] * u2_val) % mod
                
        total_sum = (total_sum + h[k]) % mod
        
    result = (total_sum * inverse(m)) % mod
    print(result)

if __name__ == '__main__':
    solve()