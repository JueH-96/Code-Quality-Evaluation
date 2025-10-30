# YOUR CODE HERE
from collections import defaultdict

N = int(input())
events = [list(map(int, input().split())) for _ in range(N)]
dp = defaultdict(int)
dp[0] = 0

for t, x in events:
    if t == 1:
        dp[x] = max(dp[x], dp[0] + 1)
    else:
        dp[x] = max(dp[x], dp[0])
        dp[0] = max(dp[0], dp[x] - 1)

if dp[0] < 0:
    print(-1)
else:
    max_potions = max(dp.values())
    actions = [0] * N
    current_potions = 0
    for i, (t, x) in enumerate(events):
        if t == 1:
            if dp[x] - 1 >= dp[0]:
                actions[i] = 1
                current_potions += 1
            dp[x] = max(dp[x], dp[0] + 1)
        else:
            if dp[x] - 1 >= dp[0]:
                current_potions -= 1
            dp[0] = max(dp[0], dp[x] - 1)

    print(max_potions)
    print(*actions)