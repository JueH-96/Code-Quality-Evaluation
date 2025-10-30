import sys

readline = sys.stdin.readline
read = sys.stdin.read
ns = lambda: readline().rstrip()
ni = lambda: int(readline().rstrip())
nm = lambda: map(int, readline().split())
nl = lambda: list(map(int, readline().split()))
prn = lambda x: print(*x, sep='
')

def solve():
    N, W = nm()
    WV = [nl() for _ in range(N)]
    WV.sort()
    dp = [-float('inf')] * (W+1)
    dp[0] = 0
    for w, v in WV:
        for i in range(W, w-1, -1):
            dp[i] = max(dp[i], dp[i-w] + v - w)
        for i in range(w, W+1):
            dp[i] = max(dp[i], dp[i-w] + v)
    print(max(dp))
    return

# solve()

t = 1
t = ni()
for _ in range(t):
    solve()