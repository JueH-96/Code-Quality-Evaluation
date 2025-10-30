import sys

# Read N and C from the first line
N, C = map(int, sys.stdin.readline().split())

# Read the list of press times T from the second line
T = list(map(int, sys.stdin.readline().split()))

# Initialize the count of candies received
candies_received = 0

# Initialize the time a candy was last received.
# For the first press, a candy is always received, so this will be T[0].
last_candy_time = -1 

# Process the first press (always receives a candy)
if N > 0: # Ensure there's at least one press
    candies_received = 1
    last_candy_time = T[0]

# Process subsequent presses
for i in range(1, N):
    current_press_time = T[i]
    
    # Calculate the time elapsed since the last candy was received
    time_since_last_candy = current_press_time - last_candy_time
    
    # Check if enough time has elapsed to receive a new candy
    if time_since_last_candy >= C:
        candies_received += 1
        # If a candy is received, update the last_candy_time to the current press time
        last_candy_time = current_press_time
    # If not enough time has elapsed, last_candy_time remains unchanged

# Print the total number of candies received
print(candies_received)