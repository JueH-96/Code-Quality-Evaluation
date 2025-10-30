import sys

def can_buy_units(products, M, target_units):
    """Check if we can buy exactly target_units with budget M"""
    products.sort()  # Sort by price
    total_cost = 0
    remaining_units = target_units
    
    for price in products:
        if remaining_units == 0:
            break
            
        # For this product, buy as many units as we can/need
        # We want to minimize cost, so we buy units one by one until we run out of remaining_units
        # But we need to be smart about it - we can buy at most remaining_units from this product
        units_from_this = min(remaining_units, int((M - total_cost) ** 0.5 / (price ** 0.5)) + 1)
        
        # Try different numbers of units from this product
        best_units = 0
        for units in range(min(remaining_units + 1, 1000)):  # Cap the search to avoid TLE
            cost = units * units * price
            if total_cost + cost <= M:
                best_units = units
            else:
                break
        
        total_cost += best_units * best_units * price
        remaining_units -= best_units
        
        if total_cost > M:
            return False
    
    return remaining_units == 0

def solve():
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    
    # Binary search on the answer
    left, right = 0, int(2 * (M ** 0.5)) + 1
    
    while left < right:
        mid = (left + right + 1) // 2
        
        # Check if we can buy mid units total
        # Greedy approach: sort products by price and buy from cheapest first
        products = sorted(P)
        total_cost = 0
        remaining = mid
        
        for price in products:
            if remaining == 0:
                break
            
            # For this product, find optimal number of units to buy
            # We want to buy as many as possible without exceeding budget
            units = min(remaining, int(((M - total_cost) / price) ** 0.5) + 1)
            
            # Check exact number
            while units > 0 and total_cost + units * units * price > M:
                units -= 1
            
            total_cost += units * units * price
            remaining -= units
            
            if total_cost > M:
                break
        
        if remaining == 0 and total_cost <= M:
            left = mid
        else:
            right = mid - 1
    
    print(left)

solve()