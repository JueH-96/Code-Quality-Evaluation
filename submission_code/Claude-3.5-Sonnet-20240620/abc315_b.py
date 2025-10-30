# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for month, days in enumerate(D, 1):
    if current_day + days >= middle_day:
        print(month, middle_day - current_day)
        break
    current_day += days