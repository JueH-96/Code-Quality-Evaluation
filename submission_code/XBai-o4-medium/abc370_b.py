n = int(input())
A = [[] for _ in range(n + 1)]  # 1-based indexing

for i in range(1, n + 1):
    A[i] = list(map(int, input().split()))

current = 1
for m in range(1, n + 1):
    x = current
    y = m
    if x >= y:
        current = A[x][y - 1]
    else:
        current = A[y][x - 1]

print(current)