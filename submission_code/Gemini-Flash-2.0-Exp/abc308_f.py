def solve():
    n, m = map(int, input().split())
    p = list(map(int, input().split()))
    l = list(map(int, input().split()))
    d = list(map(int, input().split()))

    coupons = []
    for i in range(m):
        coupons.append((l[i], d[i]))

    coupons.sort(reverse=True)

    total_price = 0
    used_coupons = [False] * m
    
    items = sorted([(price, i) for i, price in enumerate(p)], reverse=True)

    for price, item_index in items:
        best_discount = 0
        best_coupon_index = -1

        for i in range(m):
            if not used_coupons[i] and coupons[i][0] <= price and coupons[i][1] > best_discount:
                best_discount = coupons[i][1]
                best_coupon_index = i

        if best_coupon_index != -1:
            total_price += price - best_discount
            used_coupons[best_coupon_index] = True
        else:
            total_price += price

    print(total_price)

solve()