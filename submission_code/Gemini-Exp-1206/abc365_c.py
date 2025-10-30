def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    total_cost = sum(a)
    if total_cost <= m:
        print("infinite")
        return

    def check(x):
        total_subsidy = 0
        for cost in a:
            total_subsidy += min(x, cost)
        return total_subsidy <= m

    left, right = 0, 10**9 + 1
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