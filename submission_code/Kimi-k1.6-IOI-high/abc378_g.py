A, B, M = map(int, input().split())
n = A * B - 1
k = A - 1

if k < 0 or k > n:
    print(0)
else:
    max_n = n
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % M
    
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], M-2, M)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % M
    
    bin_coeff = fact[n] * inv_fact[k] % M
    bin_coeff = bin_coeff * inv_fact[n - k] % M
    print(bin_coeff)