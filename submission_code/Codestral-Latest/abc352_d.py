import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Create a dictionary to map each value to its index
index_map = {value: index for index, value in enumerate(P)}

# Initialize the minimum difference to a large value
min_diff = float('inf')

# Iterate through all possible starting points for the good index sequence
for start in range(N - K + 1):
    # Get the values in the current window of size K
    window_values = P[start:start + K]
    # Sort the values in the window
    sorted_values = sorted(window_values)
    # Check if the sorted values form a consecutive sequence
    if all(sorted_values[i] == sorted_values[0] + i for i in range(K)):
        # Calculate the difference between the last and first index in the window
        diff = index_map[sorted_values[-1]] - index_map[sorted_values[0]]
        # Update the minimum difference if the current difference is smaller
        min_diff = min(min_diff, diff)

# Print the minimum difference
print(min_diff)