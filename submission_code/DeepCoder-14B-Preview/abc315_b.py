M = int(input())
D = list(map(int, input().split()))
total = sum(D)
middle_day = (total + 1) // 2

current_sum = 0
for i in range(M):
    current_sum += D[i]
    if current_sum >= middle_day:
        sum_before = current_sum - D[i]
        day = middle_day - sum_before
        print(i + 1, day)
        break