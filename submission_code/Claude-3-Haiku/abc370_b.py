# Read the input
N = int(input())
A = [[0] * N for _ in range(N)]
for i in range(N):
    row = list(map(int, input().split()))
    for j in range(i+1):
        A[i][j] = row[j]

# Combine elements
current = 1
for i in range(1, N+1):
    if i >= current:
        current = A[current-1][i-1]
    else:
        current = A[i-1][current-1]

# Print the final element
print(current)