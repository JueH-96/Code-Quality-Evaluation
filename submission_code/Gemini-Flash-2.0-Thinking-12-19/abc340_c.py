def solve():
    n = int(input())
    memo = {}
    
    def calculate_cost(current_n):
        if current_n < 2:
            return 0
        if current_n in memo:
            return memo[current_n]
        l = current_n // 2
        r = (current_n + 1) // 2
        cost = current_n + calculate_cost(l) + calculate_cost(r)
        memo[current_n] = cost
        return cost
        
    result = calculate_cost(n)
    print(result)

if __name__ == '__main__':
    solve()