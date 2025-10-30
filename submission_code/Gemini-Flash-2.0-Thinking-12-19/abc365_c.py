def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_a_sum = sum(a)
    if total_a_sum <= m:
        print("infinite")
        return
        
    low = 0
    high = m
    max_valid_x = 0
    
    while low <= high:
        mid = (low + high) // 2
        current_subsidy_sum = 0
        for cost in a:
            current_subsidy_sum += min(mid, cost)
            
        if current_subsidy_sum <= m:
            max_valid_x = mid
            low = mid + 1
        else:
            high = mid - 1
            
    print(max_valid_x)

if __name__ == '__main__':
    solve()