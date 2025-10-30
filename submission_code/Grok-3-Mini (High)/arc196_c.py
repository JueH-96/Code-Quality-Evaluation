# YOUR CODE HERE
import sys
MOD = 998244353

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = data[index]

if S[0] != 'B' or S[2 * N - 1] != 'W':
    print(0)
else:
    ans = 1
    for i in range(1, N + 1):
        ans = (ans * i) % MOD
    print(ans)