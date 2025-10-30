def main():
    import sys, heapq
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Read items prices.
    prices = [int(next(it)) for _ in range(n)]
    # Read coupon thresholds.
    thresholds = [int(next(it)) for _ in range(m)]
    # Read coupon discounts.
    discounts = [int(next(it)) for _ in range(m)]
    
    # Sort items prices in non-decreasing order.
    prices.sort()
    
    # Prepare coupons as (L, D) tuples and sort by L (threshold)
    coupons = list(zip(thresholds, discounts))
    coupons.sort(key=lambda x: x[0])
    
    total_cost = 0
    coupon_index = 0
    # We'll use a max-heap (simulate with min-heap of negative discounts)
    valid_coupons = []
    
    # For each item (cheapest to most expensive)
    for price in prices:
        # Add all coupons that become available for the current item (price >= L)
        while coupon_index < m and coupons[coupon_index][0] <= price:
            # Push the discount (negate to simulate max-heap)
            heapq.heappush(valid_coupons, -coupons[coupon_index][1])
            coupon_index += 1
        
        # If there is any valid coupon available, use the one with the maximum discount.
        if valid_coupons:
            best_discount = -heapq.heappop(valid_coupons)
            total_cost += price - best_discount
        else:
            total_cost += price

    sys.stdout.write(str(total_cost))


if __name__ == '__main__':
    main()