# YOUR CODE HERE
# Read input
import sys
import threading

def main():
    N, Q = map(int, sys.stdin.readline().split())
    instructions = []
    for _ in range(Q):
        H_i, T_i = sys.stdin.readline().split()
        T_i = int(T_i) - 1  # zero-based indexing
        instructions.append((H_i, T_i))

    N = int(N)
    left_pos = 0  # Left hand starts at part 1 (index 0)
    right_pos = 1 % N  # Right hand starts at part 2 (index 1)
    total_moves = 0

    for H_i, T_i in instructions:
        if H_i == 'L':
            specified_hand_pos = left_pos
            other_hand_pos = right_pos
            specified_hand = 'L'
            other_hand = 'R'
        else:
            specified_hand_pos = right_pos
            other_hand_pos = left_pos
            specified_hand = 'R'
            other_hand = 'L'

        # Compute shortest distance and direction from specified hand to T_i
        dist_cw = (T_i - specified_hand_pos) % N
        dist_ccw = (specified_hand_pos - T_i) % N
        if dist_cw <= dist_ccw:
            direction = 1  # clockwise
            steps = dist_cw
        else:
            direction = -1  # counter-clockwise
            steps = dist_ccw

        # Simulate movement
        for _ in range(steps):
            next_pos = (specified_hand_pos + direction) % N
            # Check if other hand is blocking
            if next_pos == other_hand_pos:
                # Move other hand away
                # Try to move other hand in opposite direction
                other_direction = -direction
                possible_moves = []
                move1 = (other_hand_pos + other_direction) % N
                if move1 != specified_hand_pos and move1 != next_pos:
                    possible_moves.append(move1)
                move2 = (other_hand_pos + direction) % N
                if move2 != specified_hand_pos and move2 != next_pos:
                    possible_moves.append(move2)
                if possible_moves:
                    other_hand_pos = possible_moves[0]
                    total_moves += 1
                else:
                    # This should not happen as N >=3, but just in case
                    pass  # No valid move for other hand
            # Move specified hand
            specified_hand_pos = next_pos
            total_moves += 1

        # Update positions
        if specified_hand == 'L':
            left_pos = specified_hand_pos
            right_pos = other_hand_pos
        else:
            right_pos = specified_hand_pos
            left_pos = other_hand_pos

    print(total_moves)

# Run main in a thread to avoid recursion depth limitation with pypy
threading.Thread(target=main).start()