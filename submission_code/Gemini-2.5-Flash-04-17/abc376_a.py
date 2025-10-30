# Read N and C from the first line of input
n, c = map(int, input().split())

# Read the list of button press times T from the second line of input
t = list(map(int, input().split()))

# Initialize the number of candies received.
# The first press (at time t[0]) always results in a candy.
candies = 1

# Initialize the time the last candy was received.
# This is initially the time of the first press.
last_candy_time = t[0]

# Iterate through the remaining button presses starting from the second press (index 1)
for i in range(1, n):
    current_time = t[i]
    
    # Calculate the time difference since the last time a candy was received
    time_diff = current_time - last_candy_time
    
    # Check if enough time (at least C seconds) has elapsed since the last candy
    if time_diff >= c:
        # If enough time has elapsed, Takahashi receives a candy
        candies += 1
        # Update the time the last candy was received to the current press time
        last_candy_time = current_time
    # Else (if time_diff < c), less than C seconds have passed, so no candy is received
    # In this case, last_candy_time remains unchanged, referring to the time of the previous candy.

# Print the total number of candies received
print(candies)