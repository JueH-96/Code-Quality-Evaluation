# YOUR CODE HERE
N, M = map(int, input().split())
mod = 998244353
n = N * M
R = (n * (n + 1)) % mod
print(R)