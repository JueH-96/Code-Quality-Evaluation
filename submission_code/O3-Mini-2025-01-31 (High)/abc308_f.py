def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Read the prices of the items.
    prices = [int(next(it)) for _ in range(N)]
    # Read coupon eligibility thresholds L and discounts D.
    Ls = [int(next(it)) for _ in range(M)]
    Ds = [int(next(it)) for _ in range(M)]
    
    # Sort the items in increasing order.
    prices.sort()
    # Pair coupons as (threshold, discount) and sort them by threshold.
    coupons = list(zip(Ls, Ds))
    coupons.sort(key=lambda x: x[0])
    
    total_cost = 0
    coupon_index = 0
    heap = []  # We'll use this max-heap (simulated with negatives) to store discount values.
    
    # Process each item in increasing order of price.
    # For each item, add all coupons that become applicable (i.e. whose L <= price)
    # and then use the coupon with the largest discount available.
    for price in prices:
        while coupon_index < M and coupons[coupon_index][0] <= price:
            # Python's heapq is a min-heap so we push negative discount to simulate max-heap.
            heapq.heappush(heap, -coupons[coupon_index][1])
            coupon_index += 1
        if heap:
            best_discount = -heapq.heappop(heap)
            total_cost += price - best_discount
        else:
            total_cost += price
    
    sys.stdout.write(str(total_cost))


if __name__ == '__main__':
    main()