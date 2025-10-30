# YOUR CODE HERE

N = int(input())
H = list(map(int, input().split()))

A = [0] * (N + 1)

operations = [0] * N

for i in range(1, N + 1):
    A[0] += 1
    for j in range(1, i + 1):
        while A[j - 1] > A[j] and A[j - 1] > H[j - 1]:
            A[j - 1] -= 1
            A[j] += 1
            operations[j - 1] += 1

print(*operations)