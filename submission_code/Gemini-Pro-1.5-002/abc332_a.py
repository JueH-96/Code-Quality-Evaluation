# YOUR CODE HERE
n, s, k = map(int, input().split())
total_price = 0
for _ in range(n):
    p, q = map(int, input().split())
    total_price += p * q

if total_price < s:
    total_price += k

print(total_price)