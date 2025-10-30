N, R, C = map(int, input().split())
S = input().strip()

smoke_positions = {(0, 0)}
result = []

for t in range(N):
    direction = S[t]
    
    # Move all smoke according to wind
    new_positions = set()
    for r, c in smoke_positions:
        if direction == 'N':
            new_positions.add((r-1, c))
        elif direction == 'S':
            new_positions.add((r+1, c))
        elif direction == 'W':
            new_positions.add((r, c-1))
        elif direction == 'E':
            new_positions.add((r, c+1))
    
    smoke_positions = new_positions
    
    # Generate new smoke at (0,0) if none exists there
    if (0, 0) not in smoke_positions:
        smoke_positions.add((0, 0))
    
    # Check if smoke exists at (R, C)
    if (R, C) in smoke_positions:
        result.append('1')
    else:
        result.append('0')

print(''.join(result))