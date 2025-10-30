# YOUR CODE HERE
def can_buy_units(prices, M, target_units):
    """Check if we can buy at least target_units with budget M"""
    total_cost = 0
    total_units = 0
    
    for p in prices:
        # For each product, find max units we can buy
        # Binary search for k such that k^2 * p <= remaining budget
        left, right = 0, int((M / p) ** 0.5) + 1
        while left < right:
            mid = (left + right + 1) // 2
            if mid * mid * p <= M - total_cost + p * mid * mid:
                left = mid
            else:
                right = mid - 1
        
        k = left
        if k > 0:
            cost = k * k * p
            if total_cost + cost <= M:
                total_cost += cost
                total_units += k
    
    return total_units >= target_units

def solve():
    N, M = map(int, input().split())
    prices = list(map(int, input().split()))
    
    # Sort prices to prioritize cheaper products
    prices.sort()
    
    # Binary search on the answer
    left, right = 0, int((M / min(prices)) ** 0.5) * N + 1
    
    while left < right:
        mid = (left + right + 1) // 2
        
        # Check if we can buy mid units in total
        total_cost = 0
        total_units = 0
        
        # Greedy approach: buy from cheapest products first
        remaining_target = mid
        
        for p in prices:
            if remaining_target <= 0:
                break
            
            # Maximum units we can buy from this product
            max_units = int((M - total_cost) / p) ** 0.5
            units_to_buy = min(max_units, remaining_target)
            
            if units_to_buy > 0:
                cost = units_to_buy * units_to_buy * p
                if total_cost + cost <= M:
                    total_cost += cost
                    total_units += units_to_buy
                    remaining_target -= units_to_buy
        
        if total_units >= mid:
            left = mid
        else:
            right = mid - 1
    
    print(left)

solve()