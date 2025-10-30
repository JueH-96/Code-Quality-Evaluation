MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    
    # Precompute f_j(x) for all j and x
    f = [ [0]*(M+2) for _ in range(N) ]
    for j in range(N):
        if B[j] != -1:
            for x in range(M+1):
                f[j][x] = 1 if B[j] > x else 0
        else:
            for x in range(M+1):
                f[j][x] = max(0, M - x)
    
    # Precompute F_i(x) for all i and x
    F = [ [1]*(M+1) for _ in range(N+1) ]  # F[i][x] is product of f_j(x) for j <i
    for i in range(1, N+1):
        for x in range(1, M+1):
            F[i][x] = F[i-1][x] * f[i-1][x] % MOD
    
    # Precompute the number of -1s after each i
    q = [0]*N
    for i in range(N):
        cnt = 0
        for j in range(i+1, N):
            if B[j] == -1:
                cnt += 1
        q[i] = cnt
    
    ans = 0
    for i in range(N):
        # Compute sum_x in X_i of F[i][x] * M^{q[i]}
        if B[i] != -1:
            x = B[i]
            valid = F[i][x]
            ans = (ans + valid * pow(M, q[i], MOD)) % MOD
        else:
            # Sum F[i][x] for x=1..M
            total = 0
            for x in range(1, M+1):
                total = (total + F[i][x]) % MOD
            ans = (ans + total * pow(M, q[i], MOD)) % MOD
    
    print(ans % MOD)

if __name__ == '__main__':
    main()