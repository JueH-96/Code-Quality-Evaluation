def remaining_water(N, additions):
    current_water = 0
    last_time = 0
    
    for T, V in additions:
        # Calculate the time passed since the last addition
        time_passed = T - last_time
        
        # Water leaks during the time passed
        current_water = max(0, current_water - time_passed)
        
        # Add the new water
        current_water += V
        
        # Update the last time
        last_time = T
    
    return current_water

# Read input
import sys
input = sys.stdin.read
data = input().strip().splitlines()

N = int(data[0])
additions = [tuple(map(int, line.split())) for line in data[1:]]

# Calculate and print the result
result = remaining_water(N, additions)
print(result)