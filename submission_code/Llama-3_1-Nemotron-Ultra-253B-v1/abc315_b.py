M = int(input())
D = list(map(int, input().split()))
total = sum(D)
target = (total + 1) // 2
cumulative = 0
for i in range(M):
    if cumulative + D[i] >= target:
        print(i + 1, target - cumulative)
        break
    cumulative += D[i]