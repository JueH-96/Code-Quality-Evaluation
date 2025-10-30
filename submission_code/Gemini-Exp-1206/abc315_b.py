M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
middle_day = (total_days + 1) // 2
current_day = 0
for i, days in enumerate(D):
    if current_day + days >= middle_day:
        month = i + 1
        day = middle_day - current_day
        print(month, day)
        break
    current_day += days