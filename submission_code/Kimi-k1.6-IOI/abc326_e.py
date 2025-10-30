mod = 998244353

N = int(input())
A = list(map(int, input().split()))

if N == 0:
    print(0)
else:
    inv_N = pow(N, mod - 2, mod)
    
    # Precompute (N+1)^(y-1) for y from 1 to N (i.e., exponents 0 to N-1)
    pow_N_plus_1 = [1] * N
    for i in range(1, N):
        pow_N_plus_1[i] = pow_N_plus_1[i-1] * (N + 1) % mod
    
    # Precompute inv(N)^y for y from 0 to N
    inv_N_pows = [1] * (N + 1)
    for i in range(1, N + 1):
        inv_N_pows[i] = inv_N_pows[i-1] * inv_N % mod
    
    total = 0
    for y in range(1, N + 1):
        a = A[y-1]
        p = pow_N_plus_1[y-1]
        inv = inv_N_pows[y]
        term = a * p % mod
        term = term * inv % mod
        total = (total + term) % mod
    
    print(total)