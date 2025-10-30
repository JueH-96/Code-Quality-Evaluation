# YOUR CODE HERE
M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for month in range(M):
    current_day += D[month]
    if current_day >= middle_day:
        day_of_month = D[month] - (current_day - middle_day)
        print(month + 1, day_of_month)
        break