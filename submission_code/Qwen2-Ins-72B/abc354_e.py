from itertools import product

N = int(input())
AB = [tuple(map(int, input().split())) for _ in range(N)]

memo = {}

def dfs(state, turn):
    if turn >= N:
        return 0
    if state in memo:
        return memo[state]
    aoki = dfs(state, turn + 1)
    takahashi = aoki
    for i, j in product(range(N), repeat=2):
        if state & (1 << i) and state & (1 << j) and (i != j):
            if AB[i][0] == AB[j][0] or AB[i][0] == AB[j][1] or AB[i][1] == AB[j][0] or AB[i][1] == AB[j][1]:
                takahashi = max(takahashi, 1 - dfs(state ^ (1 << i) ^ (1 << j), turn + 1))
    memo[state] = max(aoki, takahashi)
    return memo[state]

state = (1 << N) - 1
result = dfs(state, 0)
if result == 0:
    print("Aoki")
else:
    print("Takahashi")