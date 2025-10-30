# YOUR CODE HERE
def simulate_smoke_movement(S, R, C):
    N = len(S)
    smoke_positions = {(0, 0)}  # Initially, smoke is at (0, 0)
    results = []
    
    for t in range(N):
        # Move the smoke according to the wind direction
        new_smoke_positions = set()
        for r, c in smoke_positions:
            if S[t] == 'N':
                new_smoke_positions.add((r-1, c))
            elif S[t] == 'W':
                new_smoke_positions.add((r, c-1))
            elif S[t] == 'S':
                new_smoke_positions.add((r+1, c))
            else:  # S[t] == 'E'
                new_smoke_positions.add((r, c+1))
        
        # After all smoke has moved, check if new smoke needs to be generated at (0, 0)
        if (0, 0) not in new_smoke_positions:
            new_smoke_positions.add((0, 0))
        
        smoke_positions = new_smoke_positions
        
        # Check if smoke exists at (R, C) at time t+0.5
        results.append('1' if (R, C) in smoke_positions else '0')
    
    return ''.join(results)

# Read input
line = input().split()
N, R, C = map(int, line)
S = input().strip()

# Simulate and print the result
result = simulate_smoke_movement(S, R, C)
print(result)