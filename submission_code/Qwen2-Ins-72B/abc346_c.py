N, K = map(int, input().split())
A = list(map(int, input().split()))

A.sort()

i = 0
ans = 0
for k in range(1, K+1):
    while i < N and A[i] < k:
        i += 1
    if i == N or A[i] != k:
        ans += k

print(ans)