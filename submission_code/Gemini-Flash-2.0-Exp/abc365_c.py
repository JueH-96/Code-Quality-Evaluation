def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    if sum(a) <= m:
        print("infinite")
        return

    low = 0
    high = int(2e9)  # A reasonable upper bound for x

    ans = 0
    while low <= high:
        mid = (low + high) // 2
        total_subsidy = 0
        for val in a:
            total_subsidy += min(mid, val)

        if total_subsidy <= m:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    print(ans)

solve()