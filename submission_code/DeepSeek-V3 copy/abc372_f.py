MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    
    adj = [[] for _ in range(N+1)]
    
    for i in range(1, N+1):
        adj[i].append((i % N) + 1)
    
    for i in range(M):
        X = int(data[3 + 2*i])
        Y = int(data[4 + 2*i])
        adj[X].append(Y)
    
    dp = [0] * (N+1)
    dp[1] = 1
    
    for _ in range(K):
        new_dp = [0] * (N+1)
        for u in range(1, N+1):
            if dp[u]:
                for v in adj[u]:
                    new_dp[v] = (new_dp[v] + dp[u]) % MOD
        dp = new_dp
    
    print(dp[1] % MOD)

if __name__ == "__main__":
    main()