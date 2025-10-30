# YOUR CODE HERE
N, C = map(int, input().split())
T = list(map(int, input().split()))
candies_received = 0
last_candy_time = None
for t in T:
    if last_candy_time is None or t - last_candy_time >= C:
        candies_received += 1
        last_candy_time = t
print(candies_received)