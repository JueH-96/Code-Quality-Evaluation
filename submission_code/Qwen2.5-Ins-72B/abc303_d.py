# YOUR CODE HERE
import sys

def min_time_to_type(X, Y, Z, S):
    # Initialize the total time and the state of the Caps Lock key
    total_time = 0
    caps_lock_on = False

    # Iterate through the string to calculate the minimum time
    for i in range(len(S)):
        if S[i] == 'A':
            if caps_lock_on:
                total_time += X
            else:
                total_time += Y
        else:  # S[i] == 'a'
            if caps_lock_on:
                total_time += Y
            else:
                total_time += X

        # Check if we need to toggle the Caps Lock key
        if i < len(S) - 1 and S[i] != S[i + 1]:
            if S[i] == 'A' and S[i + 1] == 'a' or S[i] == 'a' and S[i + 1] == 'A':
                total_time += Z
                caps_lock_on = not caps_lock_on

    return total_time

# Read input
X, Y, Z = map(int, input().split())
S = input().strip()

# Calculate and print the result
print(min_time_to_type(X, Y, Z, S))