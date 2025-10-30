# YOUR CODE HERE
def min_operations_to_follow_instructions(N, Q, instructions):
    # Initial positions of left and right hands
    left_hand = 1
    right_hand = 2
    total_operations = 0

    for instruction in instructions:
        hand, target = instruction
        if hand == 'L':
            # Move left hand to target
            if target > left_hand:
                clockwise_distance = target - left_hand
                counterclockwise_distance = N - target + left_hand
            else:
                clockwise_distance = N - left_hand + target
                counterclockwise_distance = left_hand - target
            total_operations += min(clockwise_distance, counterclockwise_distance)
            left_hand = target
        elif hand == 'R':
            # Move right hand to target
            if target > right_hand:
                clockwise_distance = target - right_hand
                counterclockwise_distance = N - target + right_hand
            else:
                clockwise_distance = N - right_hand + target
                counterclockwise_distance = right_hand - target
            total_operations += min(clockwise_distance, counterclockwise_distance)
            right_hand = target

    return total_operations

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
instructions = [(data[i*2+2], int(data[i*2+3])) for i in range(Q)]

# Calculate and print the result
result = min_operations_to_follow_instructions(N, Q, instructions)
print(result)