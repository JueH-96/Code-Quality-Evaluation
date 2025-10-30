# YOUR CODE HERE
def solve():
    import sys, math
    S = sys.stdin.read().strip()
    N = len(S)
    INF = math.inf
    dp = [INF] * (N +1)
    dp[0] =0
    buttons = {'00','0','1','2','3','4','5','6','7','8','9'}
    for i in range(1, N+1):
        for b in buttons:
            lb = len(b)
            if i >= lb and S[i-lb:i] == b:
                if dp[i - lb] +1 < dp[i]:
                    dp[i] = dp[i - lb] +1
    print(dp[N])