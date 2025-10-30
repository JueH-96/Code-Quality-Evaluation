M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
middle_day = (total_days + 1) // 2
current_sum = 0

for i in range(M):
    current_sum += D[i]
    if current_sum >= middle_day:
        a = i + 1
        b = middle_day - (current_sum - D[i])
        break

print(a, b)