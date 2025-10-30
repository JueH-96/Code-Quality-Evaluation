from collections import defaultdict

def preprocess(S, T):
    n = len(S)
    m = len(T)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    last = defaultdict(lambda: -1)
    
    for i in range(n):
        dp[i + 1][0] = 1
        last[S[i]] = i
    
    for i in range(n):
        for j in range(m):
            if S[i] == T[j]:
                dp[i + 1][j + 1] = dp[i][j]
            dp[i + 1][j] += dp[i][j]
    
    return dp, last

def solve(S, T):
    dp, last = preprocess(S, T)
    n = len(S)
    m = len(T)
    ans = 0
    
    for i in range(n):
        for j in range(m):
            if S[i] == T[j]:
                ans += dp[last[S[i]]][j] * (n - i)
    
    return ans

N, T = input().split()
S = []
for _ in range(int(N)):
    S.append(input())

print(solve(''.join(S), T))