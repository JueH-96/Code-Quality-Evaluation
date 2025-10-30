r, g, b = map(int, input().split())
c = input()
price = [r, g, b]
if c == 'Red':
    price[0] = float('inf')
elif c == 'Green':
    price[1] = float('inf')
else:
    price[2] = float('inf')
print(min(price))