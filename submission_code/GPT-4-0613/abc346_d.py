import sys
from itertools import groupby

def solve():
    N = int(input().strip())
    S = input().strip()
    C = list(map(int, input().strip().split()))
    C.append(0)
    S += '1'
    G = [list(g) for k, g in groupby(S)]
    dp = [0]*(len(G)+1)
    for i in range(len(G)-1, -1, -1):
        if G[i][0] == '0':
            dp[i] = min(dp[i+1]+min(C[j] for j in G[i]), dp[i+2]+C[G[i][-1]] if i+2<len(G) else float('inf'))
        else:
            dp[i] = dp[i+1]
    print(dp[0])

solve()