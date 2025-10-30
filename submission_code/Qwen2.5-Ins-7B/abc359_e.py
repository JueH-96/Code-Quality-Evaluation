# N = int(input())
# H = list(map(int, input().split()))

N = 5
H = [3, 1, 4, 1, 5]

A = [0] * (N + 1)
steps = []

for i in range(1, N + 1):
    steps.append((H[i - 1] + 1) * i)
    for j in range(i - 1, 0, -1):
        if A[j - 1] > A[j] and A[j - 1] > H[j]:
            A[j - 1] -= 1
            A[j] += 1

for i in range(1, N + 1):
    print(steps[i - 1], end=' ')