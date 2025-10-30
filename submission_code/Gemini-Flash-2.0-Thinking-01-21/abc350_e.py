import math

def solve():
    n, a, x, y = map(int, input().split())
    memo = {}
    
    def get_expected_cost(current_n):
        if current_n == 0:
            return 0.0
        if current_n in memo:
            return memo[current_n]
        
        cost1 = x + get_expected_cost(current_n // a)
        
        sum_costs_div_b = 0
        for b in range(2, 7):
            sum_costs_div_b += get_expected_cost(current_n // b)
        cost2 = (6/5) * y + (1/5) * sum_costs_div_b
        
        result = min(cost1, cost2)
        memo[current_n] = result
        return result
        
    expected_cost = get_expected_cost(n)
    print(f"{expected_cost:.12f}")

if __name__ == '__main__':
    solve()