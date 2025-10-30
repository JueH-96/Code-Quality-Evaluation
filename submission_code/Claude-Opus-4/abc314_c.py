# YOUR CODE HERE
N, M = map(int, input().split())
S = list(input())
C = list(map(int, input().split()))

# For each color from 1 to M
for color in range(1, M + 1):
    # Find all positions with this color
    positions = []
    for i in range(N):
        if C[i] == color:
            positions.append(i)
    
    # If there are positions with this color, perform right circular shift
    if positions:
        # Store the characters at these positions
        chars = [S[pos] for pos in positions]
        
        # Right circular shift: last element goes to first
        shifted_chars = [chars[-1]] + chars[:-1]
        
        # Put the shifted characters back
        for i, pos in enumerate(positions):
            S[pos] = shifted_chars[i]

# Print the final string
print(''.join(S))