def min_operations_to_follow_instructions(N, Q, instructions):
    # Initial positions of the hands
    left_pos = 1
    right_pos = 2
    total_moves = 0
    
    def distance_on_ring(N, start, end):
        # Calculate the minimum distance on a ring of size N from start to end
        if start <= end:
            direct = end - start
            wrap_around = start + N - end
        else:
            direct = N - start + end
            wrap_around = start - end
        return min(direct, wrap_around)
    
    for hand, target in instructions:
        if hand == 'L':
            # Move left hand to target
            moves = distance_on_ring(N, left_pos, target)
            left_pos = target
        elif hand == 'R':
            # Move right hand to target
            moves = distance_on_ring(N, right_pos, target)
            right_pos = target
        total_moves += moves
    
    return total_moves

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
Q = int(data[1])
instructions = []
index = 2
for _ in range(Q):
    H = data[index]
    T = int(data[index + 1])
    instructions.append((H, T))
    index += 2

result = min_operations_to_follow_instructions(N, Q, instructions)
print(result)