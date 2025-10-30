M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle_day = (total_days + 1) // 2

current_day = 0
for i in range(M):
    current_day += D[i]
    if current_day >= middle_day:
        month = i + 1
        day = middle_day - (current_day - D[i])
        print(month, day)
        break