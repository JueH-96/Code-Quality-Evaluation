n, p, q = map(int, input().split())
d = list(map(int, input().split()))
min_d = min(d)
option_coupon = q + min_d
print(min(p, option_coupon))