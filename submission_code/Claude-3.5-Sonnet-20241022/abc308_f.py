def solve():
    N, M = map(int, input().split())
    P = list(map(int, input().split()))
    L = list(map(int, input().split()))
    D = list(map(int, input().split()))

    # Create list of (price, index) pairs and sort by price descending
    items = [(p, i) for i, p in enumerate(P)]
    items.sort(reverse=True)
    
    # Create list of (min_price, discount, index) for coupons and sort by discount descending
    coupons = [(L[i], D[i], i) for i in range(M)]
    coupons.sort(key=lambda x: x[1], reverse=True)
    
    # Keep track of which items have been discounted
    used_items = [False] * N
    
    total_cost = sum(P)
    
    # Try to apply each coupon to the most expensive eligible item
    for min_price, discount, _ in coupons:
        # Find most expensive unused item that meets minimum price
        for price, item_idx in items:
            if price >= min_price and not used_items[item_idx]:
                total_cost -= discount
                used_items[item_idx] = True
                break
                
    print(total_cost)

solve()