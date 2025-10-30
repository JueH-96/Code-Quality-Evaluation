def min_operations(N, Q, instructions):
    # Initial positions of hands
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for H_i, T_i in instructions:
        if H_i == 'L':
            # Move left hand to T_i
            target_hand = left_hand
            target_position = T_i
            other_hand = right_hand
        else:
            # Move right hand to T_i
            target_hand = right_hand
            target_position = T_i
            other_hand = left_hand

        # Calculate the distance to move the target hand
        distance = min(
            (target_position - target_hand) % N,
            (target_hand - target_position) % N
        )
        
        # Move the other hand if necessary
        if other_hand != target_position:
            # Calculate the distance to move the other hand
            distance_other = min(
                (target_position - other_hand) % N,
                (other_hand - target_position) % N
            )
            # Move the other hand to the target position
            total_operations += distance + distance_other
        else:
            total_operations += distance

        # Update the positions of the hands
        if H_i == 'L':
            left_hand = T_i
        else:
            right_hand = T_i

    return total_operations

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, Q = map(int, data[0].split())
instructions = [line.split() for line in data[1:]]
instructions = [(H, int(T)) for H, T in instructions]

# Calculate the result
result = min_operations(N, Q, instructions)

# Print the result
print(result)