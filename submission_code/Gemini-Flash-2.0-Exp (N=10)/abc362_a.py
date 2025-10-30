r, g, b = map(int, input().split())
c = input()

prices = []
if c != "Red":
  prices.append(r)
if c != "Green":
  prices.append(g)
if c != "Blue":
  prices.append(b)

print(min(prices))