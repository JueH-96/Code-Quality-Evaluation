N, C = map(int, input().split())
T = list(map(int, input().split()))
candies = 0
last_time = -C  # Initialize to allow first press to always count

for t in T:
    if t - last_time >= C:
        candies += 1
        last_time = t

print(candies)