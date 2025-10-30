# YOUR CODE HERE
import sys

def solve():
    # Read the number of water additions
    N = int(sys.stdin.readline())

    # Initialize current water level and the time of the last event
    current_water = 0
    last_time = 0

    # Process each water addition
    for _ in range(N):
        # Read the time T_i and volume V_i for the current addition
        T_i, V_i = map(int, sys.stdin.readline().split())

        # Calculate the time elapsed since the last water addition
        time_elapsed = T_i - last_time

        # Simulate water leakage during the elapsed time
        # The amount of water cannot go below 0
        current_water = max(0, current_water - time_elapsed)

        # Add the new water to the humidifier
        current_water += V_i

        # Update the last_time to the current addition time for the next iteration
        last_time = T_i

    # Print the final amount of water remaining in the humidifier
    print(current_water)

# Call the solve function to execute the program
solve()