# YOUR CODE HERE
N, D = map(int, input().split())
S = [input() for _ in range(N)]

max_days = 0
for day in range(D):
    if all(S[i][day] == 'o' for i in range(N)):
        consecutive_days = 1
        while day + 1 < D and all(S[i][day + 1] == 'o' for i in range(N)):
            day += 1
            consecutive_days += 1
        max_days = max(max_days, consecutive_days)

print(max_days)