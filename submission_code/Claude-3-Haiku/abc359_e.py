N = int(input())
H = [int(x) for x in input().split()]

A = [0] * (N + 1)
result = []

for i in range(1, N + 1):
    operations = 0
    while A[i-1] <= max(A[i], H[i-1]):
        A[0] += 1
        for j in range(1, i):
            if A[j-1] > A[j] and A[j-1] > H[j-1]:
                A[j-1] -= 1
                A[j] += 1
        operations += 1
    result.append(operations)

print(" ".join(map(str, result)))