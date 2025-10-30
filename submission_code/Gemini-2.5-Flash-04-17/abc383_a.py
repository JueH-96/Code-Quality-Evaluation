# Read the number of water additions
N = int(input())

# Initialize the current time and water level
current_time = 0
water_level = 0

# Process each water addition event
for _ in range(N):
    # Read the time of addition and the amount of water added
    line = input().split()
    T_i = int(line[0])
    V_i = int(line[1])

    # Calculate the time elapsed since the last event
    # The first event is from time 0, subsequent events are from the previous T_i
    time_elapsed = T_i - current_time

    # Water leaks out during the elapsed time.
    # The amount leaked is equal to the time elapsed.
    # The water level cannot drop below 0.
    water_level = max(0, water_level - time_elapsed)

    # Add the new water
    water_level = water_level + V_i

    # Update the current time to the time of the current event
    current_time = T_i

# After processing all N events, the water_level holds the amount
# immediately after the last addition at time T_N.
print(water_level)