def main():
    import sys
    from collections import deque

    input = sys.stdin.readline
    # Read N (number of parts) and Q (number of queries)
    N, Q = map(int, input().split())

    # Initialize snake: the deque will store tuples (x, y) for each part.
    # The head (part 1) is at index 0, and tail (part N) is at index N-1.
    # Initially part i is at (i, 0) so head is at (1,0), part 2 at (2, 0) etc.
    snake = deque((i, 0) for i in range(1, N + 1))

    # We'll accumulate outputs for type 2 queries.
    output = []

    # Define movement directions
    moves = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }

    # Process each query.
    for _ in range(Q):
        query = input().split()
        if query[0] == '1':
            # Query type 1: move the head by 1.
            direction = query[1]
            dx, dy = moves[direction]
            # The current head is snake[0]
            head_x, head_y = snake[0]
            new_head = (head_x + dx, head_y + dy)
            # Insert the new head at the beginning and remove the tail.
            snake.appendleft(new_head)
            snake.pop()
        else:
            # Query type 2: output the position of part p.
            p = int(query[1])
            part = snake[p - 1]  # p-th part is at index p-1
            output.append(f"{part[0]} {part[1]}")

    # Write all outputs at once.
    sys.stdout.write("
".join(output))
    
if __name__ == "__main__":
    main()