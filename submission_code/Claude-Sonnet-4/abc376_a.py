# YOUR CODE HERE
n, c = map(int, input().split())
times = list(map(int, input().split()))

candies = 0
last_candy_time = -float('inf')  # Initialize to negative infinity so first press always works

for t in times:
    if t - last_candy_time >= c:
        candies += 1
        last_candy_time = t

print(candies)