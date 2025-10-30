def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    sum_a = sum(a)
    if sum_a <= m:
        print("infinite")
        return

    low = 0
    high = max(a)
    ans = 0

    while low <= high:
        mid = (low + high) // 2
        total_subsidy = sum(min(mid, val) for val in a)
        if total_subsidy <= m:
            ans = mid
            low = mid + 1
        else:
            high = mid - 1

    print(ans)

if __name__ == "__main__":
    solve()