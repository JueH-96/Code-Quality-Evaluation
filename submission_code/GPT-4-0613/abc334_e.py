MOD = 998244353
H, W = map(int, input().split())
S = [list(input()) for _ in range(H)]
fact = [1] * (H * W + 1)
inv = [1] * (H * W + 1)
for i in range(1, H * W + 1):
    fact[i] = fact[i - 1] * i % MOD
    inv[i] = pow(fact[i], MOD - 2, MOD)

def comb(n, k):
    if n < k or n < 0 or k < 0:
        return 0
    return fact[n] * inv[k] * inv[n - k] % MOD

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]
cnt = [[0] * 4 for _ in range(H * W)]
for i in range(H):
    for j in range(W):
        if S[i][j] == '#':
            continue
        for k in range(4):
            x = i + dx[k]
            y = j + dy[k]
            if x < 0 or x >= H or y < 0 or y >= W or S[x][y] == '#':
                continue
            cnt[i * W + j][k] += 1

dp = [[0] * (H * W + 1) for _ in range(H * W + 1)]
dp[0][0] = 1
for i in range(H * W):
    for j in range(i + 1):
        dp[i + 1][j] = (dp[i + 1][j] + dp[i][j] * (j + 1)) % MOD
        dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD

ans = 0
for i in range(H):
    for j in range(W):
        if S[i][j] == '#':
            continue
        for k in range(4):
            x = i + dx[k]
            y = j + dy[k]
            if x < 0 or x >= H or y < 0 or y >= W or S[x][y] == '#':
                continue
            ans = (ans + dp[i * W + j][cnt[i * W + j][k]]) % MOD
        ans = (ans + comb(H * W - 1, cnt[i * W + j][0] + cnt[i * W + j][1] + cnt[i * W + j][2] + cnt[i * W + j][3])) % MOD
print(ans)