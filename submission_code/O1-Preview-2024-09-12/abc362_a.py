R_G_B = input()
R, G, B = map(int, R_G_B.split())
C = input()

color_prices = {'Red': R, 'Green': G, 'Blue': B}
del color_prices[C]
min_price = min(color_prices.values())
print(min_price)