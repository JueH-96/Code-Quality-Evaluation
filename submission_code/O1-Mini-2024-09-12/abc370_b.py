N = int(input())
A = [[] for _ in range(N + 1)]

for i in range(1, N + 1):
    row = list(map(int, input().split()))
    A[i] = [0] + row  # 1-based indexing

current = 1

for k in range(1, N + 1):
    if current >= k:
        current = A[current][k]
    else:
        current = A[k][current]

print(current)