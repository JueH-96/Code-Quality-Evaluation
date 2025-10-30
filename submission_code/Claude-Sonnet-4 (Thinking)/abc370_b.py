n = int(input())
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

current = 1  # Starting element

for j in range(1, n + 1):  # Combine with elements 1, 2, ..., N
    if current >= j:
        current = A[current - 1][j - 1]  # A[current, j] (convert to 0-indexed)
    else:
        current = A[j - 1][current - 1]  # A[j, current] (convert to 0-indexed)

print(current)