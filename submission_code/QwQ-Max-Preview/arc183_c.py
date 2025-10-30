MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    conditions = [[] for _ in range(N+1)]  # 1-based X
    
    for _ in range(M):
        L = int(input[idx])
        R = int(input[idx+1])
        X = int(input[idx+2])
        idx += 3
        conditions[X].append((L, R))
    
    # Precompute forbidden_p for each k (element)
    forbidden_p = [set() for _ in range(N+2)]  # forbidden_p[k] is for element k
    
    for k in range(1, N+1):
        for p in range(1, N+1):
            for (L, R) in conditions[p]:
                if R <= k:
                    forbidden_p[k].add(p-1)  # convert to 0-based
    
    # DP with mask
    dp = [0] * (1 << N)
    dp[0] = 1
    
    for step in range(1, N+1):
        current_k = N - step + 1
        current_forbidden = forbidden_p[current_k]
        new_dp = [0] * (1 << N)
        for mask in range(1 << N):
            if dp[mask] == 0:
                continue
            for p in range(N):
                if (mask & (1 << p)) == 0 and p not in current_forbidden:
                    new_mask = mask | (1 << p)
                    new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
        dp = new_mask = None
        dp = new_dp
    
    # The answer is the value in dp[(1 << N) - 1]
    print(dp[(1 << N) - 1] % MOD)

if __name__ == '__main__':
    main()