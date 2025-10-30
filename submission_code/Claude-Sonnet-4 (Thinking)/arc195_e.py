def solve():
    N, Q = map(int, input().split())
    A = [0, 0] + list(map(int, input().split()))  # A[2], A[3], ..., A[N]
    
    MOD = 998244353
    
    # Precompute (N-1)!
    fact = 1
    for i in range(1, N):
        fact = (fact * i) % MOD
    
    for _ in range(Q):
        u, v = map(int, input().split())
        
        ans = 0
        
        for i in range(2, N + 1):
            if i == u:
                # C(u, u, v) = (N-1)! * (u-1) / u
                coeff = (fact * (u - 1) * pow(u, MOD - 2, MOD)) % MOD
            elif i == v:
                # C(v, u, v) = (N-1)!
                coeff = fact
            elif i < v:
                # C(i, u, v) = (N-1)! / i
                coeff = (fact * pow(i, MOD - 2, MOD)) % MOD
            else:  # i > v
                # C(i, u, v) = 0
                coeff = 0
            
            ans = (ans + A[i] * coeff) % MOD
        
        print(ans)

solve()