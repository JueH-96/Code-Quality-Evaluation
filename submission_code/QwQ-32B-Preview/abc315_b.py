M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day_position = (total_days + 1) // 2

current_day = 0
for a in range(M):
    current_day += D[a]
    if current_day >= middle_day_position:
        month = a + 1
        day_in_month = middle_day_position - (current_day - D[a])
        print(month, day_in_month)
        break