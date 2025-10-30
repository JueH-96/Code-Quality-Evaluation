# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

M = int(data[0])
D = list(map(int, data[1:]))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for month in range(M):
    current_day += D[month]
    if current_day >= middle_day:
        print(month + 1, middle_day - (current_day - D[month]))
        break