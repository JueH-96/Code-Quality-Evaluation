def solve():
    n, m = map(int, input().split())
    fireworks_days = list(map(int, input().split()))
    results = []
    for i in range(1, n + 1):
        low = 0
        high = m - 1
        first_fireworks_index = -1
        while low <= high:
            mid = (low + high) // 2
            if fireworks_days[mid] >= i:
                first_fireworks_index = mid
                high = mid - 1
            else:
                low = mid + 1
        if first_fireworks_index != -1:
            results.append(fireworks_days[first_fireworks_index] - i)
        else:
            # This case should not happen given the problem constraints, as fireworks on the last day is guaranteed.
            # However, for robustness, we might need to handle it, although it's not strictly necessary for this problem.
            pass 
            
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()