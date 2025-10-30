import sys
from itertools import product
from heapq import heapify, heappop, heappush

def main():
    N, K, P = map(int, input().split())
    C = [0]*N
    A = [[0]*K for _ in range(N)]
    for i in range(N):
        C[i], *A[i] = map(int, input().split())
    dp = [[float('inf')]*(P+1) for _ in range(K+1)]
    dp[0][0] = 0
    for i in range(N):
        ndp = [list(dp[j]) for j in range(K+1)]
        for j in range(K+1):
            for k in range(P+1):
                if dp[j][k] == float('inf'):
                    continue
                for l in range(j+1):
                    for m in range(min(P, k+A[i][l])+1):
                        ndp[max(j, l+1)][max(k, m)] = min(ndp[max(j, l+1)][max(k, m)], dp[j][k]+C[i])
        dp = ndp
    print(dp[K][P] if dp[K][P] != float('inf') else -1)

main()