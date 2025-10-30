# YOUR CODE HERE

N = int(input())
V_N = (N * (N + 1) // 2) % 998244353
print(V_N * N % 998244353)