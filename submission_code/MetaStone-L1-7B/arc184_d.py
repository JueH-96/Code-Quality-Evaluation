import bisect

MOD = 998244353

n = int(input())
x = list(map(int, input().split()))
y = list(map(int, input().split()))

balls = sorted(zip(x, y), key=lambda p: p[0])
ys = [b[1] for b in balls]

dp = [0] * n
prefix = [0] * (n + 1)

for i in range(n):
    current_y = ys[i]
    # Find the number of j < i where ys[j] >= current_y
    cnt = 0
    for j in range(i):
        if ys[j] >= current_y:
            cnt += 1
    dp[i] = 1 + sum(dp[j] for j in range(i) if ys[j] >= current_y)
    # Update prefix sum for binary search
    prefix[i + 1] = prefix[i] + dp[i]

total = sum(dp) % MOD
print(total)