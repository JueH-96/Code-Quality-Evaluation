M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

curr_days = 0
for month in range(M):
    if curr_days + D[month] >= middle_day:
        day = middle_day - curr_days
        print(month + 1, day)
        break
    curr_days += D[month]