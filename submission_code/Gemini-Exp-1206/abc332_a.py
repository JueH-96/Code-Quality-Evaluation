N, S, K = map(int, input().split())
total_price = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total_price += P * Q
if total_price >= S:
    shipping_fee = 0
else:
    shipping_fee = K
print(total_price + shipping_fee)