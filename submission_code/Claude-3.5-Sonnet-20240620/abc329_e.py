# YOUR CODE HERE
def can_match(S, T):
    N, M = len(S), len(T)
    dp = [False] * (N + 1)
    dp[0] = True

    for i in range(N + 1):
        if dp[i]:
            if i + M <= N and S[i:i+M] == T:
                dp[i+M] = True
            if i < N and S[i] == '#':
                dp[i+1] = True

    return dp[N]

N, M = map(int, input().split())
S = input().strip()
T = input().strip()

X = '#' * N
if can_match(S, T):
    print("Yes")
else:
    print("No")