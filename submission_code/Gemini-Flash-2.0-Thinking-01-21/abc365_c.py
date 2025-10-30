def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_cost_sum = sum(a)
    if total_cost_sum <= m:
        print("infinite")
    else:
        max_a = max(a)
        low = 0
        high = max_a
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            current_subsidy = 0
            for cost in a:
                current_subsidy += min(mid, cost)
            if current_subsidy <= m:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        print(ans)

if __name__ == '__main__':
    solve()