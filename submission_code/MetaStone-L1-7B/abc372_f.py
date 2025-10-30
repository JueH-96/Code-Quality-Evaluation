MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    K = int(data[idx]); idx +=1
    
    edges = [[] for _ in range(N)]
    for _ in range(M):
        X = int(data[idx]); idx +=1
        Y = int(data[idx]); idx +=1
        edges[X].append(Y)
    
    dp = [0] * N
    dp[0] = 1  # node 1 is index 0
    
    for _ in range(K):
        new_dp = [0] * N
        # Main cycle transition: shift
        for v in range(N):
            u = (v - 1) % N
            new_dp[u] = (new_dp[u] + dp[v]) % MOD
        # Extra edges
        for u in range(N):
            for v in edges[u]:
                new_dp[v] = (new_dp[v] + dp[u]) % MOD
        dp = new_dp
    
    print(dp[0] % MOD)

if __name__ == '__main__':
    main()