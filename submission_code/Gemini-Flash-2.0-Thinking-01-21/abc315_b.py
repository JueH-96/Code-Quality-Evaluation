M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day_number = (total_days + 1) // 2

cumulative_days_before_month = 0
result_month = 0
result_day = 0

for month_index in range(M):
    days_in_month = D[month_index]
    start_day_of_month = cumulative_days_before_month + 1
    end_day_of_month = cumulative_days_before_month + days_in_month
    if start_day_of_month <= middle_day_number <= end_day_of_month:
        result_month = month_index + 1
        result_day = middle_day_number - cumulative_days_before_month
        break
    cumulative_days_before_month = end_day_of_month

print(result_month, result_day)