MOD = 998244353

S = input().strip()
n = len(S)
prev = [0] * (n + 1)
prev[0] = 1

for c in S:
    curr = [0] * (n + 1)
    for j in range(n + 1):
        if prev[j] == 0:
            continue
        if c == '(':
            if j + 1 <= n:
                curr[j + 1] = (curr[j + 1] + prev[j]) % MOD
        elif c == ')':
            if j - 1 >= 0:
                curr[j - 1] = (curr[j - 1] + prev[j]) % MOD
        else:
            if j + 1 <= n:
                curr[j + 1] = (curr[j + 1] + prev[j]) % MOD
            if j - 1 >= 0:
                curr[j - 1] = (curr[j - 1] + prev[j]) % MOD
    prev = curr

print(prev[0] % MOD)