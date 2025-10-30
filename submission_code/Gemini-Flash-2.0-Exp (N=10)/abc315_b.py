m = int(input())
days = list(map(int, input().split()))

total_days = sum(days)
middle_day = (total_days + 1) // 2

current_day = 0
for i in range(m):
    if current_day + days[i] >= middle_day:
        month = i + 1
        day = middle_day - current_day
        print(month, day)
        break
    current_day += days[i]