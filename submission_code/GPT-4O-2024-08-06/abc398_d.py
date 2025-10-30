# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3]
    
    # Initialize the smoke set with the starting point (0, 0)
    smoke_positions = {(0, 0)}
    
    # Initialize the result list
    result = []
    
    # Directions mapping
    direction_map = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }
    
    # Process each time step
    for t in range(N):
        # Determine the direction of the wind at time t
        direction = S[t]
        dr, dc = direction_map[direction]
        
        # Move all smoke positions according to the wind direction
        new_smoke_positions = set()
        for r, c in smoke_positions:
            new_r = r + dr
            new_c = c + dc
            new_smoke_positions.add((new_r, new_c))
        
        # Check if smoke exists at (R, C) at time t+0.5
        if (R, C) in new_smoke_positions:
            result.append('1')
        else:
            result.append('0')
        
        # Update smoke positions
        smoke_positions = new_smoke_positions
        
        # If there's no smoke at (0, 0), generate new smoke at (0, 0)
        if (0, 0) not in smoke_positions:
            smoke_positions.add((0, 0))
    
    # Print the result as a string
    print(''.join(result))

if __name__ == "__main__":
    main()