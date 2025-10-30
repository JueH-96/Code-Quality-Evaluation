# Read N (number of presses) and C (cool-down time)
N, C = map(int, input().split())

# Read the list of press times T_1, T_2, ..., T_N
T_values = list(map(int, input().split()))

# Constraints state 1 <= N <= 100, so T_values will not be empty.

# The first press always yields a candy.
candies_received = 1
# Store the time when the last candy was received.
# Initially, it's the time of the first press.
last_candy_receipt_time = T_values[0]

# Iterate from the second press (index 1) up to the last press (index N-1).
# If N=1, range(1, 1) is empty, so this loop will not execute.
# In that case, candies_received remains 1, which is correct for a single press.
for i in range(1, N):
    current_press_time = T_values[i]
    
    # Calculate the time elapsed since the last candy was actually received.
    time_elapsed_since_last_candy = current_press_time - last_candy_receipt_time
    
    # A candy is received if C or more seconds have passed.
    # This is because the condition for *not* receiving a candy is
    # "less than C seconds have elapsed".
    if time_elapsed_since_last_candy >= C:
        candies_received += 1
        # Update the time of the last candy receipt to the current press time,
        # as a candy was just received.
        last_candy_receipt_time = current_press_time
    # Else (time_elapsed_since_last_candy < C):
    #   No candy is received.
    #   last_candy_receipt_time is NOT updated; it still refers to the time
    #   the previous candy was successfully obtained.
    
# Print the total number of candies received.
print(candies_received)