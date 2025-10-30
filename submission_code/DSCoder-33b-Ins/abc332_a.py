# YOUR CODE HERE
N, S, K = map(int, input().split())
total_price = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total_price += P * Q
shipping_fee = K if total_price < S else 0
print(total_price + shipping_fee)