# YOUR CODE HERE
MOD = 10**8

def f(x, y):
    return (x + y) % MOD

N = int(input())
A = list(map(int, input().split()))

prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % MOD

result = 0
for i in range(N):
    result = (result + (i * A[i]) % MOD) % MOD
    result = (result + (prefix_sum[N] - prefix_sum[i+1] + MOD) % MOD) % MOD

print(result)