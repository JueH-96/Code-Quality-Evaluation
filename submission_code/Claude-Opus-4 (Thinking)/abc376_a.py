# YOUR CODE HERE
# Read input
N, C = map(int, input().split())
T = list(map(int, input().split()))

# First press always gives a candy
candy_count = 1
last_candy_time = T[0]

# Process remaining presses
for i in range(1, N):
    if T[i] - last_candy_time >= C:
        candy_count += 1
        last_candy_time = T[i]

print(candy_count)