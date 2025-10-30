# YOUR CODE HERE

import sys

N, D = map(int, sys.stdin.readline().split())

schedules = [sys.stdin.readline().strip() for _ in range(N)]

max_free_days = 0
current_free_days = 0

for day in range(D):
    if all('o' == schedule[day] for schedule in schedules):
        current_free_days += 1
    else:
        max_free_days = max(max_free_days, current_free_days)
        current_free_days = 0

max_free_days = max(max_free_days, current_free_days)

print(max_free_days)