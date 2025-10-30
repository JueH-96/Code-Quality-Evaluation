# N D
N, D = map(int, input().split())
# S_i
schedules = [input() for _ in range(N)]
max_days = 0

for start in range(D):
    for end in range(start, D):
        if all(schedules[i][start:end+1] == 'o' * (end - start + 1) for i in range(N)):
            max_days = max(max_days, end - start + 1)

print(max_days)