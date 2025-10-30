# YOUR CODE HERE
N, S, K = map(int, input().split())

total_product_cost = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total_product_cost += P * Q

if total_product_cost >= S:
    shipping_fee = 0
else:
    shipping_fee = K

total_amount = total_product_cost + shipping_fee
print(total_amount)