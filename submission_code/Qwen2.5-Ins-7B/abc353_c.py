# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
mod = 10**8
prefix_sum = [0]
for i in range(n):
    prefix_sum.append((prefix_sum[-1] + a[i]) % mod)
result = 0
for i in range(1, n+1):
    for j in range(i+1, n+1):
        result += (a[i-1] + a[j-1]) % mod
        result %= mod
print(result)