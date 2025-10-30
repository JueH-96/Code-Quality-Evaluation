# YOUR CODE HERE

N, P, Q = map(int, input().split())
D = list(map(int, input().split()))

min_price = P

for i in range(N):
    total_price = D[i] + Q
    if total_price < min_price:
        min_price = total_price

print(min_price)