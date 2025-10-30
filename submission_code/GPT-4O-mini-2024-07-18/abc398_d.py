def smoke_at_time(N, R, C, S):
    # Initialize the position of smoke
    smoke_positions = {(0, 0)}
    result = []

    # Process each time step
    for t in range(N):
        # Move smoke according to the wind direction
        new_smoke_positions = set()
        for r, c in smoke_positions:
            if S[t] == 'N':
                new_smoke_positions.add((r - 1, c))
            elif S[t] == 'W':
                new_smoke_positions.add((r, c - 1))
            elif S[t] == 'S':
                new_smoke_positions.add((r + 1, c))
            elif S[t] == 'E':
                new_smoke_positions.add((r, c + 1))

        # Update smoke positions
        smoke_positions = new_smoke_positions

        # Check if smoke exists at (R, C) at time t + 0.5
        if (R, C) in smoke_positions:
            result.append('1')
        else:
            result.append('0')

        # If there's no smoke at (0, 0), generate new smoke
        if (0, 0) not in smoke_positions:
            smoke_positions.add((0, 0))

    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, R, C = map(int, data[0].split())
S = data[1].strip()

# Get the result
output = smoke_at_time(N, R, C, S)

# Print the output
print(output)