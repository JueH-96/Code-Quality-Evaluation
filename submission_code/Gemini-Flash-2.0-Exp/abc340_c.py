def solve():
    n = int(input())
    
    def calculate_cost(x):
        if x < 2:
            return 0
        return x + calculate_cost(x // 2) + calculate_cost((x + 1) // 2)
    
    print(calculate_cost(n))

solve()