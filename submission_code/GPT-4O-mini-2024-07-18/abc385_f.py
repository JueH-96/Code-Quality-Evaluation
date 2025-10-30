def max_height_not_visible(N, buildings):
    # Sort buildings by their x-coordinates (though they should already be sorted)
    buildings.sort()
    
    # Initialize the maximum height that cannot see all buildings
    max_height = -1.0
    
    # Check visibility from the left side (x = 0)
    for i in range(N):
        x_i, h_i = buildings[i]
        
        # Calculate the height at which building i is just visible
        if x_i > 0:
            height_needed = (h_i * x_i) / x_i  # This simplifies to h_i
            max_height = max(max_height, height_needed)
    
    # Check if all buildings can be seen from height 0
    if max_height < 0:
        return -1.0
    
    return max_height

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
buildings = [tuple(map(int, line.split())) for line in data[1:N+1]]

# Calculate the result
result = max_height_not_visible(N, buildings)

# Print the result with the required precision
print(f"{result:.18f}")