def simulate_step(blocks, W):
    # Convert blocks to set of positions for efficient lookup
    positions = {(x, y) for x, y in blocks}
    
    # Check if bottom row is complete
    bottom_y = min(y for _, y in positions)
    bottom_blocks = sum(1 for x, y in positions if y == bottom_y)
    
    if bottom_blocks == W:
        # Remove bottom row
        positions = {(x, y) for x, y in positions if y > bottom_y}
    
    # Sort blocks by y-coordinate (bottom to top)
    blocks_list = sorted([(x, y) for x, y in positions], key=lambda p: p[1])
    
    # Process falling
    new_positions = set()
    for x, y in blocks_list:
        if y == 1 or (x, y-1) in new_positions:
            new_positions.add((x, y))
        else:
            new_positions.add((x, y-1))
            
    return new_positions

def get_block_position(blocks, time):
    # Returns position of each block at given time
    positions = {i+1: (x, y) for i, (x, y) in enumerate(blocks)}
    current_positions = {i: pos for i, pos in positions.items()}
    
    # Simulate until stable state
    prev_state = None
    steps = 0
    cycle_start = None
    cycle_length = None
    states_history = []
    
    while steps < time and cycle_length is None:
        pos_set = {pos for pos in current_positions.values()}
        new_pos_set = simulate_step(pos_set, W)
        
        # Map blocks to their new positions
        new_positions = {}
        old_to_new = {}
        for old_pos in pos_set:
            if old_pos not in new_pos_set:
                continue
            if old_pos in new_pos_set:
                new_pos = old_pos
            else:
                new_pos = (old_pos[0], old_pos[1] - 1)
            old_to_new[old_pos] = new_pos
            
        for block_id, pos in current_positions.items():
            if pos in old_to_new:
                new_positions[block_id] = old_to_new[pos]
                
        current_state = frozenset((i, pos) for i, pos in new_positions.items())
        
        if current_state in states_history:
            cycle_start = states_history.index(current_state)
            cycle_length = steps - cycle_start
            break
            
        states_history.append(current_state)
        current_positions = new_positions
        steps += 1
    
    # If we found a cycle, fast forward
    if cycle_length is not None:
        if time > steps:
            remaining_steps = (time - cycle_start) % cycle_length
            current_state = states_history[cycle_start + remaining_steps - 1]
            current_positions = dict(current_state)
    
    return current_positions

# Read input
N, W = map(int, input().split())
blocks = []
for _ in range(N):
    x, y = map(int, input().split())
    blocks.append((x, y))

Q = int(input())
queries = []
for _ in range(Q):
    t, a = map(int, input().split())
    queries.append((t, a))

# Process each query
for time, block_id in queries:
    positions = get_block_position(blocks, time)
    print("Yes" if block_id in positions else "No")