N, M = map(int, input().split())
A = list(map(int, input().split()))

ans = [0] * N
for i in range(M-1):
    ans[A[i]-1] = 1
    ans[A[i]] = 0
    if A[i+1] - A[i] > 1:
        for j in range(A[i]+1, A[i+1]):
            ans[j-1] = j - A[i]

for i in range(N):
    print(ans[i])