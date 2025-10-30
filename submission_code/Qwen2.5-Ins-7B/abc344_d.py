# YOUR CODE HERE
from collections import defaultdict

def solve():
    T = input().strip()
    N = int(input().strip())
    bags = [defaultdict(int) for _ in range(N)]
    for i in range(N):
        A = int(input().strip())
        for _ in range(A):
            s = input().strip()
            bags[i][s] += 1

    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0

    for i in range(len(T)):
        for j in range(i + 1):
            if dp[j] != float('inf'):
                for k, v in bags:
                    if T[j:i+1] in v:
                        dp[i+1] = min(dp[i+1], dp[j] + (1 if i+1 < len(T) else 0))

    ans = dp[-1] if dp[-1] != float('inf') else -1
    print(ans)

solve()