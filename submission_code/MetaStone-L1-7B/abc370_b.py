n = int(input())
A = []
for _ in range(n):
    row = list(map(int, input().split()))
    A.append(row)

current = 1
for k in range(1, n + 1):
    if current >= k:
        current = A[current - 1][k - 1]
    else:
        current = A[k - 1][current - 1]

print(current)