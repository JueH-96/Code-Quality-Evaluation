# YOUR CODE HERE
import sys

# Read input N, C from the first line
line1 = sys.stdin.readline().split()
N = int(line1[0])
C = int(line1[1])

# Read input T list from the second line
# The times are guaranteed to be sorted in strictly increasing order.
T = list(map(int, sys.stdin.readline().split()))

candy_count = 0
# Initialize the time of the last candy reception.
# We set it to a value such that the condition for receiving the first candy is always met.
# Let the time of the first press be T[0]. We need T[0] - last_receive_time >= C.
# If we set last_receive_time = -C, then T[0] - (-C) = T[0] + C.
# Since T[0] >= 0 and C >= 1, T[0] + C >= 1.
# The condition becomes T[0] + C >= C, which is always true as T[0] >= 0.
# Therefore, initializing last_receive_time = -C ensures the first press grants a candy.
last_receive_time = -C 

# Iterate through each time Takahashi presses the button
for current_press_time in T:
    # Check if at least C seconds have passed since the last candy was received
    if current_press_time - last_receive_time >= C:
        # If enough time has passed, Takahashi receives a candy
        candy_count += 1
        # Update the time of the last candy reception to the current press time
        last_receive_time = current_press_time
        
# Print the total number of candies received
print(candy_count)