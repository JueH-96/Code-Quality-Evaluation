def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    candidates = sorted(list(set(a + [val + 1 for val in b])))

    for x in candidates:
        sellers_can_sell = sum(1 for price in a if price <= x)
        buyers_can_buy = sum(1 for price in b if price >= x)
        if sellers_can_sell >= buyers_can_buy:
            print(x)
            return

solve()