# YOUR CODE HERE
def max_currency(N, A, S, T):
    dp = [0] * (N + 1)
    dp[N] = A[N-1]
    
    for i in range(N-1, 0, -1):
        exchanges = A[i-1]
        j = 0
        while exchanges >= S[i-1]:
            exchanges -= S[i-1]
            j += 1
            dp[i] = max(dp[i], exchanges + min(j * T[i-1], dp[i+1]))
        dp[i] = max(dp[i], exchanges + dp[i+1])
    
    return dp[1]

N = int(input())
A = list(map(int, input().split()))
S, T = [], []
for _ in range(N-1):
    s, t = map(int, input().split())
    S.append(s)
    T.append(t)

print(max_currency(N, A, S, T))