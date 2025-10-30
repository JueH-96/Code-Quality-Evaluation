# YOUR CODE HERE
N = int(input())
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = [0] + list(map(int, input().split()))

current = 1
for i in range(1, N + 1):
    if current >= i:
        current = A[i][current]
    else:
        current = A[current][i]

print(current)