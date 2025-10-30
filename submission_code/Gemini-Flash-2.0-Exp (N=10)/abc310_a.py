def solve():
    n, p, q = map(int, input().split())
    d = list(map(int, input().split()))
    
    min_cost_with_coupon = float('inf')
    for dish_price in d:
        min_cost_with_coupon = min(min_cost_with_coupon, q + dish_price)
    
    print(min(p, min_cost_with_coupon))

solve()