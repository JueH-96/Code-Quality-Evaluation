def min_total_cost(N, M, prices, L, D):
    # Create a list of tuples for coupons with (L_i, D_i)
    coupons = sorted(zip(L, D), key=lambda x: x[0])
    
    # Sort prices in ascending order
    prices.sort()
    
    total_cost = 0
    coupon_index = 0
    available_coupons = []
    
    for price in prices:
        # Add all applicable coupons to the available_coupons list
        while coupon_index < M and coupons[coupon_index][0] <= price:
            available_coupons.append(coupons[coupon_index])
            coupon_index += 1
        
        # If we have available coupons, find the best one to use
        if available_coupons:
            # Use the coupon with the maximum discount
            best_discount = max(available_coupons, key=lambda x: x[1])[1]
            total_cost += price - best_discount
            
            # Remove the used coupon from available_coupons
            available_coupons.remove((coupons[coupon_index - len(available_coupons)][0], best_discount))
        else:
            # No coupon can be used, pay the full price
            total_cost += price
            
    return total_cost

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
prices = list(map(int, data[2:2 + N]))
L = list(map(int, data[2 + N:2 + N + M]))
D = list(map(int, data[2 + N + M:2 + N + 2 * M]))

result = min_total_cost(N, M, prices, L, D)
print(result)