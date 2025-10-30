# YOUR CODE HERE
MOD = 998244353

N = int(input())
A = list(map(int, input().split()))

ans = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        ans = (ans + int(str(A[i]) + str(A[j]))) % MOD

print(ans)