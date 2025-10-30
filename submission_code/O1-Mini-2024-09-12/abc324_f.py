# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from sys import stdin
    import math
    def input():
        return sys.stdin.read()
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M = int(data[idx]); idx +=1
    edges = [[] for _ in range(N+1)]
    for _ in range(M):
        u = int(data[idx]); idx +=1
        v = int(data[idx]); idx +=1
        b = int(data[idx]); idx +=1
        c = int(data[idx]); idx +=1
        edges[u].append( (v, b, c) )
    INF = -1e20
    def feasible(R):
        dp = [INF]*(N+1)
        dp[1] = 0.0
        for u in range(1, N+1):
            if dp[u] < -1e19:
                continue
            for v, b, c in edges[u]:
                potential = dp[u] + b - R * c
                if potential > dp[v]:
                    dp[v] = potential
        return dp[N] >= -1e-9
    low = 0.0
    high = 1e5
    for _ in range(100):
        mid = (low + high) /2
        if feasible(mid):
            low = mid
        else:
            high = mid
    print("{0:.16f}".format(low))
    
if __name__ == "__main__":
    main()