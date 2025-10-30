M = int(input())
D = list(map(int, input().split()))
total_days = sum(D)
target = (total_days + 1) // 2

cumulative = 0
a, b = 0, 0

for i in range(M):
    days = D[i]
    if cumulative + days >= target:
        a = i + 1
        b = target - cumulative
        break
    cumulative += days

print(a, b)