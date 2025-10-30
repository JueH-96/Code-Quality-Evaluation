mod = 998244353

n = int(input())
a = list(map(int, input().split()))

result = 1
for i in range(n):
    current_i = i + 1
    ai = a[i]
    result = (result * (current_i - ai)) % mod

print(result)