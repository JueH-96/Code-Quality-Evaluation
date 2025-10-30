def solve():
    n, w_max = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    dp = [-float('inf')] * (w_max + 1)
    dp[0] = 0

    for item_w, item_v in items:
        for current_w in range(w_max, -1, -1):
            for k in range(1, current_w // item_w + 1):
                weight_k = k * item_w
                happiness_k = k * item_v - k * k
                if current_w - weight_k >= 0 and dp[current_w - weight_k] != -float('inf'):
                    dp[current_w] = max(dp[current_w], dp[current_w - weight_k] + happiness_k)

    print(max(0, dp[w_max]))

solve()