def min_operations(N, instructions):
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for hand, target in instructions:
        if hand == 'L':
            current, other = left_hand, right_hand
        else:  # hand == 'R'
            current, other = right_hand, left_hand

        # If the hand is already at the target, no operation needed
        if current == target:
            continue

        # Calculate the lengths of the two paths
        forward_distance = (target - current + N) % N
        backward_distance = (current - target + N) % N

        # Check if the other hand is on the path
        other_on_forward_path = 0 < (other - current + N) % N < forward_distance
        other_on_backward_path = 0 < (current - other + N) % N < backward_distance

        # Determine the number of operations
        if other_on_forward_path:
            operations = backward_distance
        elif other_on_backward_path:
            operations = forward_distance
        else:
            operations = min(forward_distance, backward_distance)

        total_operations += operations

        # Update hand position
        if hand == 'L':
            left_hand = target
        else:  # hand == 'R'
            right_hand = target

    return total_operations

# Read inputs
N, Q = map(int, input().split())
instructions = []
for _ in range(Q):
    hand, target = input().split()
    target = int(target)
    instructions.append((hand, target))

# Calculate and print the result
print(min_operations(N, instructions))