MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    X = list(map(int, input[idx:idx+M]))
    idx += M
    
    if M == 1:
        print(0)
        return
    
    # Compute set B: elements in X[0..M-2]
    B = set()
    for i in range(M-1):
        B.add(X[i])
    len_B = len(B)
    
    # Compute set C: elements not in B
    C = []
    for c in range(1, K+1):
        if c not in B:
            C.append(c)
    r = len(C)
    
    from math import comb
    
    def compute_f(A, N, M):
        # dp[i][j]: number of sequences of length i with max prefix j
        dp = [[0] * M for _ in range(N+1)]
        dp[0][0] = 1
        for i in range(N):
            for j in range(M):
                current = dp[i][j]
                if current == 0:
                    continue
                # Stay at j
                add = current * (A - 1)
                dp[i+1][j] = (dp[i+1][j] + add) % MOD
                # Move to j+1
                if j + 1 < M:
                    dp[i+1][j+1] = (dp[i+1][j+1] + current) % MOD
        return dp[N][M-1]
    
    total = 0
    for s in range(0, r+1):
        ways = comb(r, s)
        sign = (-1)**s
        A = len_B + s
        f = compute_f(A, N, M)
        contrib = ways * sign * f
        total += contrib
        total %= MOD
    
    # Adjust for MOD
    if total < 0:
        total += MOD
    
    # Multiply by (-1)^|C|
    sign_C = (-1)**r
    total = (total * sign_C) % MOD
    
    print(total)

if __name__ == '__main__':
    main()