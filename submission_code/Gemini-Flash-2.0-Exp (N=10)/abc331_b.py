def solve():
    n, s, m, l = map(int, input().split())
    
    min_cost = float('inf')
    
    for num_s in range(n // 6 + 2):
        for num_m in range(n // 8 + 2):
            for num_l in range(n // 12 + 2):
                total_eggs = num_s * 6 + num_m * 8 + num_l * 12
                if total_eggs >= n:
                    cost = num_s * s + num_m * m + num_l * l
                    min_cost = min(min_cost, cost)
    
    print(min_cost)

solve()