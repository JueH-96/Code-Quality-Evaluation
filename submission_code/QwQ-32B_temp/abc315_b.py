# Read input
M = int(input())
D = list(map(int, input().split()))

total_days = sum(D)
middle = (total_days + 1) // 2

current = 0
for i in range(M):
    days = D[i]
    if current + days >= middle:
        a = i + 1
        b = middle - current
        print(a, b)
        break
    else:
        current += days