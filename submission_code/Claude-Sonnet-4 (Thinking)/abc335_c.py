n, q = map(int, input().split())

# History of head positions
# head_positions[0] = initial position = (1, 0)
# head_positions[i] = position of head after i-th move
head_positions = [(1, 0)]

# Direction mappings
directions = {
    'R': (1, 0),
    'L': (-1, 0),
    'U': (0, 1),
    'D': (0, -1)
}

move_count = 0

for _ in range(q):
    query = input().split()
    
    if query[0] == '1':
        # Move head
        direction = query[1]
        dx, dy = directions[direction]
        current_head = head_positions[-1]
        new_head = (current_head[0] + dx, current_head[1] + dy)
        head_positions.append(new_head)
        move_count += 1
    
    else:
        # Query part position
        p = int(query[1])
        
        if move_count >= p - 1:
            # Part p is at head_positions[move_count - (p - 1)]
            index = move_count - (p - 1)
            pos = head_positions[index]
        else:
            # Part p is at initial position of Part (p - move_count)
            pos = (p - move_count, 0)
        
        print(pos[0], pos[1])