# YOUR CODE HERE
n, x, y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(n)]
dishes.sort(key=lambda dish: dish[0] + dish[1])
dp = [0] * (1 << n)
for mask in range(1, 1 << n):
    for i in range(n):
        if mask & (1 << i):
            dp[mask] = max(dp[mask], dp[mask ^ (1 << i)] + 1)
            if (sum(d[0] for i, d in enumerate(dishes) if mask & (1 << i)) > x) or (sum(d[1] for i, d in enumerate(dishes) if mask & (1 << i)) > y):
                dp[mask] = max(dp[mask], dp[mask ^ (1 << i)])
print(dp[(1 << n) - 1])