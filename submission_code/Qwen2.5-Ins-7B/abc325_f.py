# YOUR CODE HERE
import sys
from heapq import heappush, heappop

input = sys.stdin.readline

def solve():
    N = int(input())
    D = list(map(int, input().split()))
    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())
    
    if L1 > L2:
        L1, C1, K1, L2, C2, K2 = L2, C2, K2, L1, C1, K1
    
    dp = [float('inf')] * (max(D) + 1)
    dp[0] = 0
    
    for d in D:
        for i in range(d, 0, -1):
            if i - L1 >= 0:
                dp[i] = min(dp[i], dp[i - L1] + C1)
            if i - L2 >= 0:
                dp[i] = min(dp[i], dp[i - L2] + C2)
    
    ans = min(dp[d] for d in D)
    
    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

solve()