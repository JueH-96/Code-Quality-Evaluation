# YOUR CODE HERE
n, p, q = map(int, input().split())
d = list(map(int, input().split()))

min_dish = min(d)
coupon_price = q + min_dish

print(min(p, coupon_price))