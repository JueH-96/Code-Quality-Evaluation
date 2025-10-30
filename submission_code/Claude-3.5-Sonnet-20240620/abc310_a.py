# YOUR CODE HERE
N, P, Q = map(int, input().split())
dishes = list(map(int, input().split()))

min_dish_price = min(dishes)
discounted_price = Q + min_dish_price

print(min(P, discounted_price))