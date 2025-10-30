# YOUR CODE HERE
n, m = map(int, input().split())
colors = input().split()
prices = dict()
p0, *rest = map(int, input().split())
for i in range(m):
    prices[rest[i]] = i + 1
prices['default'] = p0
total_price = sum(prices[color] for color in colors)
print(total_price)