def solve():
    n, m = map(int, input().split())
    fireworks_days = list(map(int, input().split()))
    results = []
    for i in range(1, n + 1):
        low = 0
        high = m - 1
        next_fireworks_day_index = -1
        while low <= high:
            mid = (low + high) // 2
            if fireworks_days[mid] >= i:
                next_fireworks_day_index = mid
                high = mid - 1
            else:
                low = mid + 1
        if next_fireworks_day_index != -1:
            results.append(fireworks_days[next_fireworks_day_index] - i)
        else:
            # This case should not happen given the problem constraints (A_M = N)
            results.append(0) 
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()