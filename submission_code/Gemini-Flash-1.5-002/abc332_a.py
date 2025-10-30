# YOUR CODE HERE
n, s, k = map(int, input().split())
total_price = 0
for i in range(n):
    p, q = map(int, input().split())
    total_price += p * q
if total_price >= s:
    print(total_price)
else:
    print(total_price + k)