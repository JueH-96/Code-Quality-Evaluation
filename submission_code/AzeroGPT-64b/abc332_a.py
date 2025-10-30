n, s, k = map(int, input().split())

total_price = 0
for _ in range(n):
    p, q = map(int, input().split())
    total_price += p * q

if total_price >= s:
    shipping_fee = 0
else:
    shipping_fee = k

print(total_price + shipping_fee)