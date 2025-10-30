def solve():
    MOD = 998244353
    N, K = map(int, input().split())
    
    # For N=1, black ball stays at position 1
    if N == 1:
        print(1)
        return
        
    # Calculate probability of staying at same position in one swap
    stay = (N + N*(N-1)//2) / (N*N)  # (N + C(N,2)) / N^2
    move = 1 - stay
    
    # Calculate probability after K swaps using matrix exponentiation
    def mat_mul(A, B):
        n = len(A)
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD
        return C
    
    def mat_pow(A, k):
        if k == 0:
            n = len(A)
            res = [[0]*n for _ in range(n)]
            for i in range(n):
                res[i][i] = 1
            return res
        if k == 1:
            return A
        if k % 2 == 0:
            half = mat_pow(A, k//2)
            return mat_mul(half, half)
        return mat_mul(A, mat_pow(A, k-1))
    
    # Build transition matrix
    P = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                P[i][j] = stay
            else:
                P[i][j] = move/(N-1)
                
    # Convert probabilities to fractions in mod
    for i in range(N):
        for j in range(N):
            num = int(P[i][j] * MOD)
            den = pow(1, -1, MOD)
            P[i][j] = (num * den) % MOD
            
    # Calculate final probabilities
    P_k = mat_pow(P, K)
    
    # Calculate expected value
    exp = 0
    for pos in range(N):
        exp = (exp + ((pos + 1) * P_k[0][pos])) % MOD
        
    print(exp)

solve()