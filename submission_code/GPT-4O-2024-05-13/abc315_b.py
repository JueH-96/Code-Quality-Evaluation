# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
M = int(data[0])
D = list(map(int, data[1:]))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for i in range(M):
    if current_day + D[i] >= middle_day:
        month = i + 1
        day = middle_day - current_day
        print(month, day)
        break
    current_day += D[i]