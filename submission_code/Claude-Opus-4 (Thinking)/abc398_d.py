# YOUR CODE HERE
N, R, C = map(int, input().split())
S = input().strip()

smoke = {(0, 0)}
result = []

for t in range(1, N+1):
    direction = S[t-1]
    new_smoke = set()
    
    # Move all smoke based on wind direction
    for r, c in smoke:
        if direction == 'N':
            new_smoke.add((r-1, c))
        elif direction == 'W':
            new_smoke.add((r, c-1))
        elif direction == 'S':
            new_smoke.add((r+1, c))
        elif direction == 'E':
            new_smoke.add((r, c+1))
    
    smoke = new_smoke
    
    # If no smoke at campfire, generate new smoke
    if (0, 0) not in smoke:
        smoke.add((0, 0))
    
    # Check if smoke exists at target position
    if (R, C) in smoke:
        result.append('1')
    else:
        result.append('0')

print(''.join(result))