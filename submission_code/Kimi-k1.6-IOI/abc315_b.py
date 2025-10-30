M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
target = (total_days + 1) // 2
prev_sum = 0

for i in range(M):
    if prev_sum + D[i] >= target:
        print(i + 1, target - prev_sum)
        break
    else:
        prev_sum += D[i]