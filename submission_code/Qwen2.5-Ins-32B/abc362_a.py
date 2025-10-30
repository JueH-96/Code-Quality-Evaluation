R, G, B = map(int, input().split())
C = input().strip()

prices = {'Red': R, 'Green': G, 'Blue': B}
del prices[C]

print(min(prices.values()))