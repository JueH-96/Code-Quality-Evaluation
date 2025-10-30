n = int(input())
A = [[] for _ in range(n + 1)]
for i in range(1, n + 1):
    A[i] = list(map(int, input().split()))

current = 1
for j in range(1, n + 1):
    if current >= j:
        current = A[current][j - 1]
    else:
        current = A[j][current - 1]

print(current)