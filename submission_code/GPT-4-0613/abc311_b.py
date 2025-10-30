N, D = map(int, input().split())
S = [input() for _ in range(N)]

all_free = [all(S[i][j] == 'o' for i in range(N)) for j in range(D)]

max_days = 0
current_days = 0
for free in all_free:
    if free:
        current_days += 1
        max_days = max(max_days, current_days)
    else:
        current_days = 0

print(max_days)