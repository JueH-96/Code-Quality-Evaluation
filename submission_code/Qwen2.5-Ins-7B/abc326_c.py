# N M
# A_1 A_2 ... A_N
N, M = map(int, input().split())
A = sorted(map(int, input().split()))

max_gifts = 0
for i in range(N):
    if i + M > A[i]:
        max_gifts = max(max_gifts, sum(A[j] < A[i] + M for j in range(i, N)))

print(max_gifts)