import sys

def min_operations(N, Q, instructions):
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for H, T in instructions:
        if H == 'L':
            current_hand = left_hand
            other_hand = right_hand
        else:
            current_hand = right_hand
            other_hand = left_hand

        # Calculate the shortest path to move the current hand to T
        if T == current_hand:
            continue
        elif T == other_hand:
            total_operations += 1
            current_hand, other_hand = other_hand, current_hand
        else:
            clockwise_distance = (T - current_hand) % N
            counter_clockwise_distance = (current_hand - T) % N
            shortest_distance = min(clockwise_distance, counter_clockwise_distance)
            total_operations += shortest_distance
            current_hand = T

        if H == 'L':
            left_hand = current_hand
            right_hand = other_hand
        else:
            right_hand = current_hand
            left_hand = other_hand

    return total_operations

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
Q = int(data[1])
instructions = []
for i in range(Q):
    H = data[2 * i + 2]
    T = int(data[2 * i + 3])
    instructions.append((H, T))

# Calculate and print the result
result = min_operations(N, Q, instructions)
print(result)