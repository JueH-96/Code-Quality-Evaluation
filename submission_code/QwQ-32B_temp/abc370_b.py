n = int(input())
A = [[0] * (n + 1) for _ in range(n + 1)]

for i in range(1, n + 1):
    lst = list(map(int, input().split()))
    for j in range(1, i + 1):
        A[i][j] = lst[j - 1]

current = 1
for k in range(1, n + 1):
    a = current
    b = k
    if a >= b:
        current = A[a][b]
    else:
        current = A[b][a]

print(current)