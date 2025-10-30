N = int(input())
A = list(map(int, input().split()))
cum_A = [0] + A[:]
s = sum(A)
for x in range(N):
    cum_A[x + 1] += cum_A[x]
m = s // N
M = m + 1
ans = float("inf")
for x in [m, M]:
    j = 0
    for i in range(N + 1):
        while j < N and cum_A[j + 1] < (j + 1) * x:
            j += 1
        if i <= j:
            ans = min(ans, (i * x - cum_A[i]) + (cum_A[N] - cum_A[j] - (N - j) * x))
print(ans)