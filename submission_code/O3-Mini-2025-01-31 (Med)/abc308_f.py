def main():
    import sys, heapq

    data = sys.stdin.read().split()
    if not data:
        return
    itr = iter(data)
    n = int(next(itr))
    m = int(next(itr))
    
    # Read item prices
    prices = [int(next(itr)) for _ in range(n)]
    # Read coupon thresholds L
    coupon_thresholds = [int(next(itr)) for _ in range(m)]
    # Read coupon discounts D
    coupon_discounts = [int(next(itr)) for _ in range(m)]
    
    # Create coupon tuples (L, D) and sort them by threshold L ascending
    coupons = [(coupon_thresholds[i], coupon_discounts[i]) for i in range(m)]
    coupons.sort(key=lambda x: x[0])
    
    # Sort items by price ascending
    prices.sort()
    
    total_discount = 0
    # Use a max-heap (in Python, we use a min-heap with negative values)
    heap = []
    
    # Pointer for coupons list
    j = 0
    
    # For each item, add all coupons that are now available (i.e. coupon.L <= price)
    # then assign the coupon with the greatest discount if available.
    for price in prices:
        # Add all coupons that can be applied to this item.
        while j < m and coupons[j][0] <= price:
            # Push the discount (as negative value) into the heap
            heapq.heappush(heap, -coupons[j][1])
            j += 1
        # If we have any eligible coupon, use the one with maximum discount.
        if heap:
            best_discount = -heapq.heappop(heap)
            total_discount += best_discount
    
    # The total cost is the sum of item prices minus the total discount achieved.
    total_cost = sum(prices) - total_discount
    sys.stdout.write(str(total_cost))


if __name__ == '__main__':
    main()