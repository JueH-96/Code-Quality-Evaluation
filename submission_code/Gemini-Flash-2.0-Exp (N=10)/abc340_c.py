def solve():
    n = int(input())
    
    def calculate_cost(x):
        if x < 2:
            return 0
        
        cost = x
        cost += calculate_cost(x // 2)
        cost += calculate_cost((x + 1) // 2)
        return cost
    
    print(calculate_cost(n))

solve()