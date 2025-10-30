import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx +=1
    M = int(input[idx])
    idx +=1
    
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(input[idx])
        idx +=1
        v = int(input[idx])
        idx +=1
        b = int(input[idx])
        idx +=1
        c = int(input[idx])
        idx +=1
        adj[u].append( (v, b, c) )
    
    low = 0.0
    high = 1e14
    for _ in range(100):
        mid = (low + high) / 2
        dp = [-float('inf')] * (N+1)
        dp[1] = 0.0
        for u in range(1, N+1):
            if dp[u] == -float('inf'):
                continue
            for (v, b, c) in adj[u]:
                if dp[v] < dp[u] + (b - mid * c):
                    dp[v] = dp[u] + (b - mid * c)
        if dp[N] >= 0:
            low = mid
        else:
            high = mid
    print("{0:.15f}".format(low))

if __name__ == '__main__':
    main()