n, W = map(int, input().split())
items = [tuple(map(int, input().split())) for _ in range(n)]

INF = float('-inf')
dp = [INF] * (W + 1)
dp[0] = 0

for w, v in items:
    for j in range(W, -1, -1):
        if dp[j] != INF:
            remaining = W - j
            if remaining < 0:
                continue
            t_max = remaining // w
            t_floor = v // 2
            t_ceil = (v + 1) // 2
            candidates = {t_floor, t_ceil, t_max}
            for t in candidates:
                if t < 0 or t > t_max:
                    continue
                new_j = j + t * w
                if new_j > W:
                    continue
                new_value = dp[j] + t * (v - t)
                if new_value > dp[new_j]:
                    dp[new_j] = new_value

print(max(dp))