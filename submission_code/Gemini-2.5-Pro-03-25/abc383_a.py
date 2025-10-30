import sys

# Read the number of water additions from standard input
n = int(sys.stdin.readline())

# Initialize the state variables
# current_water: the amount of water currently in the humidifier (in liters)
# current_time: the time of the last event (either start time 0 or the time of the last addition)
current_water = 0
current_time = 0

# Iterate through each of the N water addition events
for _ in range(n):
    # Read the time (t) and volume (v) for the current addition event
    # Input format is "T_i V_i" on a single line, separated by space
    line = sys.stdin.readline().split()
    t = int(line[0]) # Time of the i-th addition
    v = int(line[1]) # Volume of water added at time t

    # Calculate the time elapsed since the last event (previous addition or start time 0)
    # Since T_i are strictly increasing, time_diff will always be non-negative.
    # Specifically, time_diff > 0 because T_i < T_{i+1} implies T_i > T_{i-1} for i > 1,
    # and T_1 >= 1 > current_time=0 initially.
    time_diff = t - current_time

    # Calculate the amount of water remaining after leakage during the time interval [current_time, t).
    # The humidifier leaks 1 liter per unit time as long as there is water inside.
    # The amount of water decreases by 'time_diff' liters over the interval,
    # but it cannot drop below zero.
    # We use max(0, ...) to handle the case where the humidifier runs out of water.
    current_water = max(0, current_water - time_diff)

    # Add the volume of water (v) from the current addition event to the humidifier
    current_water += v

    # Update the current time to the time of this addition event
    # This time will be used as the starting point for calculating the time difference
    # in the next iteration (or it's the final time if this is the last addition).
    current_time = t

# After processing all N additions, print the final amount of water in the humidifier.
# The problem asks for the amount immediately after the N-th addition at time T_N.
print(current_water)