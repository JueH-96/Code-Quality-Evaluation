M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
target = (total_days + 1) // 2

current_sum = 0
for i in range(M):
    current_sum += D[i]
    if current_sum >= target:
        a = i + 1
        previous = current_sum - D[i]
        b = target - previous
        print(a, b)
        break