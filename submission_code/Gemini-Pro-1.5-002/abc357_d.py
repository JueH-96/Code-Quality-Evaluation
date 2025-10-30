# YOUR CODE HERE
n = int(input())
mod = 998244353
s = str(n)
l = len(s)
ans = 0
p = 1
for i in range(l):
    ans = (ans + int(s[i]) * p) % mod
    p = (p * 10) % mod
p = (p * pow(10, n % (mod - 1) * l, mod) - 1) * pow(pow(10, l, mod) - 1, mod - 2, mod) % mod
ans = ans * p % mod
print(ans)