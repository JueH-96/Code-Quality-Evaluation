n = int(input())
matrix = [[] for _ in range(n + 1)]  # Using 1-based indexing

for i in range(1, n + 1):
    row = list(map(int, input().split()))
    matrix[i] = row

current = 1
for k in range(1, n + 1):
    if current >= k:
        current = matrix[current][k - 1]
    else:
        current = matrix[k][current - 1]

print(current)