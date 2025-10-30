M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
middle = (total_days + 1) // 2
current_sum = 0
for i in range(M):
    if current_sum + D[i] >= middle:
        a = i + 1
        b = middle - current_sum
        print(a, b)
        break
    current_sum += D[i]