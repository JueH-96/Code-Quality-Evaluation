import sys

def solve():
    n, a, x, y = map(int, sys.stdin.readline().split())
    memo = {}
    
    def get_cost(current_n):
        if current_n == 0:
            return 0.0
        if current_n in memo:
            return memo[current_n]
        
        cost1 = x + get_cost(current_n // a)
        
        sum_costs_dice = 0
        for b in range(2, 7):
            sum_costs_dice += get_cost(current_n // b)
        cost2 = (6 * y + sum_costs_dice) / 5.0
        
        result = min(cost1, cost2)
        memo[current_n] = result
        return result
        
    min_expected_cost = get_cost(n)
    print(f"{min_expected_cost:.9f}")

if __name__ == '__main__':
    solve()