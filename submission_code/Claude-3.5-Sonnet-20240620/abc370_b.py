# YOUR CODE HERE
N = int(input())
A = [[0] * (i + 1) for i in range(N)]

for i in range(N):
    row = list(map(int, input().split()))
    for j in range(i + 1):
        A[i][j] = row[j]

current = 1
for i in range(N):
    if current <= i:
        current = A[i][current - 1]
    else:
        current = A[current - 1][i]

print(current)