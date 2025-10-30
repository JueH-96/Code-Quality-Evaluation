# YOUR CODE HERE
MOD = 998244353

N = int(input())
sN = str(N)
lenN = len(sN)
ans = 0
pw = 1
for i in range(lenN):
    ans = (ans + int(sN[lenN - 1 - i]) * pw) % MOD
    pw = (pw * (10 ** lenN + MOD -1) ) % MOD

print(ans)