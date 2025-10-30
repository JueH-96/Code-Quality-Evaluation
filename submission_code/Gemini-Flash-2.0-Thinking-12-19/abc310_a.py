def solve():
    n, p, q = map(int, input().split())
    d = list(map(int, input().split()))

    min_d = min(d)
    cost_coupon = q + min_d
    cost_regular = p

    print(min(cost_regular, cost_coupon))

solve()