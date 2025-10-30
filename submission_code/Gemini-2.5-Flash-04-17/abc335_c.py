import sys

def solve():
    # Read N and Q
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    Q = int(line1[1])

    # Store head positions history.
    # head_history[0] is the initial position of the head (1, 0).
    # head_history[i] for i > 0 is the position of the head after the i-th move.
    # The length of head_history is always (number of moves) + 1.
    head_history = [(1, 0)]

    # The number of type 1 queries processed so far.
    # This is equivalent to the number of moves the head has made.
    moves_count = 0

    for _ in range(Q):
        query = sys.stdin.readline().split()
        query_type = int(query[0])

        if query_type == 1:
            # Type 1 query: Move head by 1 in direction C
            direction = query[1]

            # Get the head's current position (after moves_count moves)
            current_head_pos = head_history[-1]
            hx, hy = current_head_pos

            # Calculate the new head position after the move
            if direction == 'R':
                new_head_pos = (hx + 1, hy)
            elif direction == 'L':
                new_head_pos = (hx - 1, hy)
            elif direction == 'U':
                new_head_pos = (hx, hy + 1)
            elif direction == 'D':
                new_head_pos = (hx, hy - 1)
            # No else needed, as C is guaranteed to be one of R, L, U, D

            # Add the new head position to the history
            head_history.append(new_head_pos)
            moves_count += 1

        elif query_type == 2:
            # Type 2 query: Find the coordinates of part p
            p = int(query[1])

            # Let k = moves_count be the number of type 1 queries processed so far.
            # head_history has length k+1, storing positions H_0, H_1, ..., H_k.
            # H_i is the head's position after i moves.

            # The position of part p after k moves is determined as follows:
            # If k >= p - 1: Part p's current position is where the head was after k - (p - 1) moves.
            #   This is H_{k - (p - 1)}. The index in head_history is k - (p - 1).
            #   This index is valid because k >= p - 1 implies k - p + 1 >= 0.
            #   Also, p >= 1 implies p - 1 >= 0, so k - (p-1) <= k.
            # If k < p - 1: The movement wave from the head hasn't reached part p yet.
            #   Part p's position is the initial position of part (p - k), which is (p - k, 0).
            #   Since k < p - 1, p - k > 1, so p - k is a valid conceptual initial part index >= 2.

            if moves_count >= p - 1:
                # Case 1: Position derived from head history
                # Index in head_history is moves_count - (p - 1)
                history_index = moves_count - p + 1
                pos = head_history[history_index]
                print(pos[0], pos[1])
            else:
                # Case 2: Position based on initial relative position
                # Position is the initial position of part p - moves_count
                pos_x = p - moves_count
                pos_y = 0 # All initial positions are on the x-axis
                print(pos_x, pos_y)

# Execute the solve function
solve()