# Read input
N, C = map(int, input().split())
T = list(map(int, input().split()))

# Initialize variables
candies = 0
last_candy_time = -float('inf')  # Time when last candy was received

# Process each button press
for i in range(N):
    current_time = T[i]
    
    # First press always gives candy, or if enough time has passed since last candy
    if current_time - last_candy_time >= C:
        candies += 1
        last_candy_time = current_time

# Output result
print(candies)