N = int(input())
A = list(map(int, input().split()))

MOD = 998244353
ans = 0

for i in range(N-1):
    for j in range(i+1, N):
        # Convert numbers to strings and concatenate
        concat = int(str(A[i]) + str(A[j]))
        ans = (ans + concat) % MOD

print(ans)