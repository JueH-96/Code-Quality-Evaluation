import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to 0
index = 0

# Read N from the first element
N = int(data[index])
index += 1

# Dictionary to store the minimum deliciousness for each color
min_per_color = {}

# Iterate through each bean
for _ in range(N):
    A = int(data[index])  # Deliciousness
    C = int(data[index + 1])  # Color
    index += 2
    
    # Update the minimum deliciousness for the color
    if C not in min_per_color:
        min_per_color[C] = A
    else:
        min_per_color[C] = min(min_per_color[C], A)

# Find the maximum of the minimum deliciousness values
max_min_del = max(min_per_color.values())

# Print the result
print(max_min_del)