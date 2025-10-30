M = int(input())
days_in_month = list(map(int, input().split()))

total_days = sum(days_in_month)
middle_day_index = (total_days + 1) // 2

cumulative_days = 0
for month_index in range(M):
    days = days_in_month[month_index]
    if middle_day_index <= cumulative_days + days:
        month_number = month_index + 1
        day_of_month = middle_day_index - cumulative_days
        print(month_number, day_of_month)
        break
    else:
        cumulative_days += days