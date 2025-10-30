# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
M = int(input[0])
D = list(map(int, input[1:]))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for month in range(M):
    if current_day + D[month] >= middle_day:
        print(month + 1, middle_day - current_day)
        break
    current_day += D[month]