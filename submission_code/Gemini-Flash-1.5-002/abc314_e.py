import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, sys.stdin.readline().split()))
        wheels.append((line[0], line[1], line[2:]))

    expected_values = []
    for i in range(n):
        c, p, s = wheels[i]
        expected_value = sum(s) / p
        expected_values.append((expected_value, c, i))

    dp = {}
    dp[0] = 0

    for i in range(1, m + 1):
        dp[i] = float('inf')
        for ev, c, idx in expected_values:
            if i - ev >= 0:
                dp[i] = min(dp[i], dp[i - int(ev)] + c)

    print(dp[m])


solve()