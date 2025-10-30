# YOUR CODE HERE
N, C = map(int, input().split())
T = list(map(int, input().split()))
candies = 0
last_candy_time = -C  # Initialize to a time that ensures the first press always gets a candy

for t in T:
    if t - last_candy_time >= C:
        candies += 1
        last_candy_time = t

print(candies)