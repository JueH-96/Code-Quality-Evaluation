def solve():
    n, x = map(int, input().split())
    processes = []
    for _ in range(n):
        processes.append(list(map(int, input().split())))

    def check(capacity):
        dp = [0] * (x + 1)
        for a, p, b, q in processes:
            for cost in range(x, -1, -1):
                for num_s in range(capacity // a + 1):
                    cost_s = num_s * p
                    if cost_s > cost:
                        break
                    remaining_capacity = capacity - num_s * a
                    num_t = remaining_capacity // b if remaining_capacity > 0 else 0
                    cost_t = num_t * q
                    total_cost = cost_s + cost_t
                    if total_cost <= cost:
                        dp[cost] = max(dp[cost], dp[cost - total_cost] + 1)
        return dp[x] >= n

    left, right = 0, 10**9 + 7
    ans = 0
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            left = mid + 1
        else:
            right = mid - 1
    print(ans)

solve()