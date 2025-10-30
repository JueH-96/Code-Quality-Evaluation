def min_operations(N, Q, instructions):
    # Initial positions of the hands
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for H_i, T_i in instructions:
        if H_i == 'L':
            # Move left hand to T_i
            # Calculate the distance in both directions
            distance_clockwise = (T_i - left_hand) % N
            distance_counterclockwise = (left_hand - T_i) % N
            # Choose the minimum distance
            move_distance = min(distance_clockwise, distance_counterclockwise)
            total_operations += move_distance
            # Update the position of the left hand
            left_hand = T_i
        else:  # H_i == 'R'
            # Move right hand to T_i
            # Calculate the distance in both directions
            distance_clockwise = (T_i - right_hand) % N
            distance_counterclockwise = (right_hand - T_i) % N
            # Choose the minimum distance
            move_distance = min(distance_clockwise, distance_counterclockwise)
            total_operations += move_distance
            # Update the position of the right hand
            right_hand = T_i

    return total_operations

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# First line contains N and Q
N, Q = map(int, data[0].split())

# Next Q lines contain the instructions
instructions = []
for i in range(1, Q + 1):
    H_i, T_i = data[i].split()
    T_i = int(T_i)
    instructions.append((H_i, T_i))

# Calculate the result
result = min_operations(N, Q, instructions)

# Print the result
print(result)