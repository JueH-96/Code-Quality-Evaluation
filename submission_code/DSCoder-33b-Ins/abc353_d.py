# YOUR CODE HERE

def f(x, y):
    return int(str(x) + str(y))

MOD = 998244353
N = int(input())
A = list(map(int, input().split()))

result = 0
for i in range(N):
    for j in range(i+1, N):
        result += f(A[i], A[j])
        result %= MOD

print(result)