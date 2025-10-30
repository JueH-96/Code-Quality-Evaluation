def solve():
    n, d, p = map(int, input().split())
    f = list(map(int, input().split()))
    
    total_fare = sum(f)
    
    min_cost = total_fare
    
    for num_passes in range(n // d + 2):
        pass_cost = num_passes * p
        
        days_covered = num_passes * d
        
        remaining_days_cost = 0
        if days_covered < n:
            remaining_days_cost = sum(sorted(f, reverse=True)[days_covered:])
        
        
        cost = pass_cost + remaining_days_cost
        min_cost = min(min_cost, cost)
        
    print(min_cost)

solve()