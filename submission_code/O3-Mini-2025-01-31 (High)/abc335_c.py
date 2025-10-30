def main():
    import sys
    # Use sys.stdin.buffer for fast I/O.
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # The first two tokens are N and Q
    N = int(data[0])
    Q = int(data[1])
    
    # Create an array to represent the snake's parts.
    # positions[i] will hold the (x, y) coordinate of the part that is at order i,
    # where the chain order is given by starting at head_index and then using modular arithmetic.
    # Initially part i (1-indexed) is at (i, 0); thus we store:
    # positions[0] is part1 (head), positions[1] is part2, ..., positions[N-1] is part N.
    positions = [(i + 1, 0) for i in range(N)]
    
    # head_index will point to the index in positions that represents the head.
    head_index = 0
    
    # Map direction bytes to coordinate delta.
    dirmap = {b"R": (1, 0), b"L": (-1, 0), b"U": (0, 1), b"D": (0, -1)}
    
    output_lines = []
    ptr = 2  # pointer to the next token from "data"
    for _ in range(Q):
        typ = data[ptr]
        ptr += 1
        if typ == b"1":
            # Move query. Next token is the direction character.
            direction = data[ptr]
            ptr += 1
            dx, dy = dirmap[direction]
            # Compute the new head position.
            cur_head_x, cur_head_y = positions[head_index]
            new_head = (cur_head_x + dx, cur_head_y + dy)
            # Instead of shifting the entire snake, we simply "rotate" our circular array:
            # new head becomes at index (head_index - 1) mod N.
            head_index = (head_index - 1) % N
            positions[head_index] = new_head
        else:
            # Query type 2: asked for the coordinates of part p.
            # p is 1-indexed and relative to the current head.
            p = int(data[ptr])
            ptr += 1
            # Calculate the index in positions corresponding to part p.
            index = (head_index + (p - 1)) % N
            x, y = positions[index]
            output_lines.append(f"{x} {y}")
            
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()