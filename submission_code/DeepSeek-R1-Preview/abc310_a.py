n, p, q = map(int, input().split())
d = list(map(int, input().split()))
min_d = min(d)
coupon_cost = q + min_d
print(min(p, coupon_cost))