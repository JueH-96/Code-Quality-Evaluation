# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))

    mod = 998244353

    N = len(A)
    ans = [0]*(N+1)
    ans[1] = N  # Any single element is an arithmetic sequence of length 1

    from collections import defaultdict

    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]

    for i in range(N):
        for j in range(i):
            d = A[i] - A[j]
            for k in dp[j][d]:
                dp[i][d][k+1] = (dp[i][d][k+1] + dp[j][d][k]) % mod
            dp[i][d][2] = (dp[i][d][2] + 1) % mod  # The pair (A[j], A[i])

    for i in range(N):
        for d in dp[i]:
            for k in dp[i][d]:
                ans[k] = (ans[k] + dp[i][d][k]) % mod

    print(' '.join(str(ans[k]%mod) for k in range(1,N+1)))

threading.Thread(target=main).start()