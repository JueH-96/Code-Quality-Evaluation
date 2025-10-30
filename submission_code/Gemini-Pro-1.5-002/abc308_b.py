# YOUR CODE HERE
n, m = map(int, input().split())
c = input().split()
d = input().split()
p = list(map(int, input().split()))
price_map = {}
for i in range(m):
    price_map[d[i]] = p[i + 1]
total_price = 0
for color in c:
    if color in price_map:
        total_price += price_map[color]
    else:
        total_price += p[0]
print(total_price)