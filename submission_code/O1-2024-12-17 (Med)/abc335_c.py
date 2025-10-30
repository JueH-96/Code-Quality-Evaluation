def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    # Read N, Q
    N = int(input_data[0])
    Q = int(input_data[1])

    # A list to store the positions of the head after each move.
    # head_positions[k] = (x, y) of the head after k moves.
    head_positions = [(1, 0)]  # Initially, after 0 moves, head is at (1,0).
    moves = 0  # How many move-queries (type 1) we have processed.

    # Precompute direction deltas
    dir_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1),
    }

    # We'll collect output for type-2 queries
    output = []
    idx = 2  # Index in input_data where queries start

    for _ in range(Q):
        query_type = int(input_data[idx])
        idx += 1

        if query_type == 1:
            # Move query
            C = input_data[idx]
            idx += 1
            dx, dy = dir_map[C]
            old_x, old_y = head_positions[moves]
            new_x, new_y = old_x + dx, old_y + dy
            head_positions.append((new_x, new_y))
            moves += 1

        else:
            # query_type == 2
            p = int(input_data[idx])
            idx += 1
            if moves >= p - 1:
                # The part p has caught up with the head's trace
                head_index = moves - (p - 1)
                x, y = head_positions[head_index]
                output.append(f"{x} {y}")
            else:
                # Part p remains in its initial position (p, 0)
                output.append(f"{p} 0")

    print("
".join(output))

# Do not forget to call main()
if __name__ == "__main__":
    main()