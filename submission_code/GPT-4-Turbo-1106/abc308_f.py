def min_total_amount(N, M, prices, limits, discounts):
    # Sort the coupons by the discount in descending order
    coupons = sorted(zip(limits, discounts), key=lambda x: -x[1])
    # Sort the prices in ascending order
    prices.sort()
    
    total = 0
    coupon_index = 0
    for price in prices:
        # Use the coupon with the largest discount for which the item is eligible
        while coupon_index < M and (coupons[coupon_index][0] > price or coupons[coupon_index][1] == 0):
            coupon_index += 1
        if coupon_index < M:
            total += price - coupons[coupon_index][1]
            coupons[coupon_index] = (coupons[coupon_index][0], 0)  # Mark the coupon as used
        else:
            total += price
    return total

# Read input
N, M = map(int, input().split())
prices = list(map(int, input().split()))
limits = list(map(int, input().split()))
discounts = list(map(int, input().split()))

# Calculate and print the answer
print(min_total_amount(N, M, prices, limits, discounts))