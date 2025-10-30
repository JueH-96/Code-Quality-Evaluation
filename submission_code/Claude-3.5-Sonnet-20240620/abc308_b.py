# YOUR CODE HERE
N, M = map(int, input().split())
plates = input().split()
colors = input().split()
prices = list(map(int, input().split()))

color_to_price = {color: price for color, price in zip(colors, prices[1:])}
default_price = prices[0]

total_price = sum(color_to_price.get(plate, default_price) for plate in plates)

print(total_price)