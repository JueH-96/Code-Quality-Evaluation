M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day_num = (total_days + 1) // 2

cumulative_days = 0
for i in range(M):
    cumulative_days += D[i]
    if cumulative_days >= middle_day_num:
        month = i + 1
        day = middle_day_num - (cumulative_days - D[i])
        print(month, day)
        break