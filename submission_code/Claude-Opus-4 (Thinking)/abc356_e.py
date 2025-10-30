N = int(input())
A = list(map(int, input().split()))

total = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        total += max(A[i], A[j]) // min(A[i], A[j])

print(total)