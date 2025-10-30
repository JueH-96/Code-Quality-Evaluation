N = int(input())
cards = [list(map(int, input().split())) for _ in range(N)]

dp = [0] * (1 << (N * 2))
sides = dict()

for i in range(N * 2):
    sides[i] = set()

for i, card in enumerate(cards):
    sides[2 * i].add(card[0])
    sides[2 * i + 1].add(card[1])
    sides[2 * i].add(card[1])
    sides[2 * i + 1].add(card[0])

def dfs(state):
    if dp[state] != 0:
        return dp[state]

    for i in range(N * 2):
        if state & (1 << i) == 0:
            continue

        for j in range(i + 1, N * 2):
            if state & (1 << j) == 0:
                continue

            if len(sides[i] & sides[j]) > 0:
                next_state = state ^ (1 << i) ^ (1 << j)
                if dfs(next_state) == 2:
                    dp[state] = 1
                    return 1
    dp[state] = 2
    return 2

if dfs((1 << (N * 2)) - 1) == 1:
    print("Takahashi")
else:
    print("Aoki")