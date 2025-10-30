MOD = 998244353
N = int(input())
A = list(map(int, input().split()))

def f(x, y):
    return int(str(x) + str(y))

total = 0
for i in range(N):
    for j in range(i+1, N):
        total = (total + f(A[i], A[j])) % MOD

print(total)