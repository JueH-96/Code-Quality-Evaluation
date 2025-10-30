N, S, K = map(int, input().split())
total_price = 0
for _ in range(N):
    P, Q = map(int, input().split())
    total_price += P * Q
if total_price >= S:
    print(total_price)
else:
    print(total_price + K)