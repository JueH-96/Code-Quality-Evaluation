# YOUR CODE HERE

# Read the inputs
inputs = input().split()
S_1, S_2 = inputs[0]
T_1, T_2 = inputs[1]

# Define the points of the pentagon
points = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# Calculate the distances
distance_S = abs(points[S_1] - points[S_2])
distance_T = abs(points[T_1] - points[T_2])

# Check if the distances are equal
if distance_S == distance_T:
    print('Yes')
else:
    print('No')