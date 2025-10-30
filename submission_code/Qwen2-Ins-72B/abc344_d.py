from collections import defaultdict

def solve():
    T = input().strip()
    N = int(input())
    dp = {T: 0}
    for _ in range(N):
        A = int(input())
        strings = input().split()[1:]
        new_dp = defaultdict(lambda: float('inf'))
        for s in strings:
            for t, cost in dp.items():
                if t.startswith(s):
                    new_dp[t[len(s):]]. = min(new_dp[t[len(s):]], cost + 1)
        dp = new_dp
    return dp[''] if dp[''] < float('inf') else -1

print(solve())