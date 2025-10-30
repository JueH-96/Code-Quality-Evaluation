import math
n, *A = map(int, open(0).read().split())
mem = [0] * 10050
for i in range(1, len(mem)):
    mem[i] = mem[i - 1] + (0.9)**i
mx, *dp, dp_ = -float("inf"), [0] * 5005
for a in A:
    for j in range(n, 0, -1):
        dp[j] = dp[j - 1]
        if j - 1 >= 0:
            x = mem[j] - mem[j - 1]
            dp[j] = max(dp[j], dp[j - 1] + a * x - 1200 / math.sqrt(j))
    dp_[0] = mem[n] * a - 1200 / math.sqrt(n + 1)
    for j in range(1, n + 1):
        x = mem[j + 1] - mem[j]
        dp_[j] = max(dp_[j - 1], dp[j - 1] + a * x - 1200 / math.sqrt(j + 1))
    mx = max(mx, max(dp[:j + 2]), dp_[j])
print(mx)