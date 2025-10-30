# Read the input
N, C = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the time of the last candy received
last_candy_time = -1

# Initialize the count of candies received
candies = 0

# Iterate through the button presses
for t in T:
    # Check if at least C seconds have elapsed since the last candy
    if t - last_candy_time >= C:
        # Receive a candy
        candies += 1
        # Update the time of the last candy received
        last_candy_time = t

# Print the output
print(candies)