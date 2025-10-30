M = int(input())
days = list(map(int, input().split()))

total_days = sum(days)
middle_day = (total_days + 1) // 2

current_day = 0
for month in range(M):
    if current_day + days[month] >= middle_day:
        # The middle day is in this month
        day_in_month = middle_day - current_day
        print(month + 1, day_in_month)
        break
    current_day += days[month]