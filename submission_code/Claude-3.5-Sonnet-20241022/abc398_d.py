def simulate_smoke(N, R, C, S):
    # Keep track of smoke positions
    smoke = {(0, 0)}
    result = []
    
    # Process each wind direction
    for t in range(N):
        # Move all existing smoke according to wind direction
        new_smoke = set()
        for r, c in smoke:
            if S[t] == 'N':
                new_smoke.add((r-1, c))
            elif S[t] == 'S':
                new_smoke.add((r+1, c))
            elif S[t] == 'E':
                new_smoke.add((r, c+1))
            else:  # 'W'
                new_smoke.add((r, c-1))
        
        # Add new smoke at campfire if none exists
        if (0, 0) not in new_smoke:
            new_smoke.add((0, 0))
            
        # Update smoke positions
        smoke = new_smoke
        
        # Check if smoke exists at target position
        result.append('1' if (R, C) in smoke else '0')
    
    return ''.join(result)

# Read input
N, R, C = map(int, input().split())
S = input().strip()

# Print result
print(simulate_smoke(N, R, C, S))