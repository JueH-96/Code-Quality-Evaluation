def solve():
    n, m = map(int, input().split())
    prices = list(map(int, input().split()))
    limits = list(map(int, input().split()))
    discounts = list(map(int, input().split()))

    coupons = []
    for i in range(m):
        coupons.append((limits[i], discounts[i]))
    
    coupons.sort(key=lambda x: x[1], reverse=True)
    
    prices_with_indices = []
    for i in range(n):
        prices_with_indices.append((prices[i], i))
    
    prices_with_indices.sort(reverse=True)
    
    used_coupons = [False] * m
    used_items = [False] * n
    total_cost = 0
    
    for price, item_index in prices_with_indices:
        best_coupon_index = -1
        
        for i in range(m):
            if not used_coupons[i] and coupons[i][0] <= price:
                if best_coupon_index == -1:
                    best_coupon_index = i
                
        if best_coupon_index != -1:
            total_cost += price - coupons[best_coupon_index][1]
            used_coupons[best_coupon_index] = True
            used_items[item_index] = True
        else:
            total_cost += price
            
    print(total_cost)

solve()