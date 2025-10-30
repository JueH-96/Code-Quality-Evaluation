# YOUR CODE HERE
n = int(input())
A = [[0] * (n+1) for _ in range(n+1)]
for i in range(1, n+1):
    A[i][i] = int(input())
    for j in range(i+1, n+1):
        A[i][j] = A[j][i] = int(input().split()[j-1-i])
current = 1
for i in range(1, n+1):
    current = A[current][i]
print(current)