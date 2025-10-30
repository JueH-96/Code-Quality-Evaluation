def main():
    import sys
    data = sys.stdin.buffer.read().split()
    iter_data = iter(data)
    N = int(next(iter_data))
    Q = int(next(iter_data))
    
    # The snake is represented as a circular list of positions.
    # Initially, part i (1-indexed) is at (i, 0).
    # We store the snake in order: index 0 is part1, index 1 is part2, …, index N-1 is partN.
    snake = [(i + 1, 0) for i in range(N)]
    head_index = 0  # head is stored at snake[head_index]
    
    # For move queries, we use this dictionary to map directions to (dx, dy)
    directions = {b'R': (1, 0), b'L': (-1, 0), b'U': (0, 1), b'D': (0, -1)}
    
    # We'll accumulate answers for type 2 queries in a list.
    output = []
    for _ in range(Q):
        op = next(iter_data)
        if op == b'1':
            # Move query. Read the direction.
            d = next(iter_data)
            dx, dy = directions[d]
            # Get current head position.
            hx, hy = snake[head_index]
            # Compute new head position.
            new_head = (hx + dx, hy + dy)
            # The snake moves like a shift: new head is inserted at the front,
            # and each part takes the previous part's position.
            # In our circular list, we simply move the head index one step backwards.
            head_index = (head_index - 1) % N
            snake[head_index] = new_head
        else:
            # Query type 2: report the coordinates of part p.
            p = int(next(iter_data))
            # The p-th part in the snake is at index (head_index + p - 1) mod N.
            x, y = snake[(head_index + p - 1) % N]
            output.append(f"{x} {y}")
    
    # Write all results.
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()