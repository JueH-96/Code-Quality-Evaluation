# YOUR CODE HERE
MOD = 998244353

def f(x, y):
    return int(str(x) + str(y))

N = int(input())
A = list(map(int, input().split()))

result = 0
power_of_10 = [1] * 11
for i in range(1, 11):
    power_of_10[i] = (power_of_10[i-1] * 10) % MOD

suffix_sum = [0] * (N + 1)
for i in range(N - 1, -1, -1):
    suffix_sum[i] = (suffix_sum[i + 1] + A[i]) % MOD

for i in range(N - 1):
    x = A[i]
    x_len = len(str(x))
    
    for j in range(i + 1, N):
        y = A[j]
        y_len = len(str(y))
        
        result = (result + f(x, y)) % MOD
        
        contribution = (x * power_of_10[y_len] + y) % MOD
        result = (result + contribution) % MOD
    
    result = (result - x * suffix_sum[i + 1] * (N - i - 1)) % MOD
    result = (result + x * (N - i - 1) * power_of_10[x_len]) % MOD

print(result)