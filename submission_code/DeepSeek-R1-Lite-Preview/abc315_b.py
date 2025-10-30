M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle = (total_days + 1) // 2

cumulative = 0
for a in range(1, M + 1):
    cumulative += D[a - 1]
    if cumulative >= middle:
        day = middle - (cumulative - D[a - 1])
        print(a, day)
        break