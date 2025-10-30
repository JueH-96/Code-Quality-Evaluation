def solve():
    n, m = map(int, input().split())
    prices = list(map(int, input().split()))
    limits = list(map(int, input().split()))
    discounts = list(map(int, input().split()))

    coupons = sorted(zip(discounts, limits, range(m)), key=lambda x: x[0], reverse=True)
    item_indices = list(range(n))
    item_indices.sort(key=lambda i: prices[i], reverse=True)

    final_prices = list(prices)
    coupon_used = [False] * m
    item_coupon_map = [-1] * n

    for discount, limit, coupon_index in coupons:
        best_item_index = -1
        best_item_price = -1

        for i in range(n):
            if item_coupon_map[i] == -1 and prices[i] >= limit:
                if best_item_index == -1 or prices[i] > best_item_price:
                    best_item_index = i
                    best_item_price = prices[i]

        if best_item_index != -1:
            final_prices[best_item_index] -= discount
            item_coupon_map[best_item_index] = coupon_index

    print(sum(final_prices))

solve()