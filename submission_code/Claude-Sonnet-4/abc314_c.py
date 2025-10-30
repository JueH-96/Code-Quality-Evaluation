# YOUR CODE HERE
N, M = map(int, input().split())
S = list(input().strip())
C = list(map(int, input().split()))

# For each color from 1 to M
for color in range(1, M + 1):
    # Find all positions with this color
    positions = []
    for i in range(N):
        if C[i] == color:
            positions.append(i)
    
    # If there's only one character of this color, no shift needed
    if len(positions) <= 1:
        continue
    
    # Get the characters at these positions
    chars = [S[pos] for pos in positions]
    
    # Perform right circular shift by 1
    # Last character moves to front, others shift right
    shifted_chars = [chars[-1]] + chars[:-1]
    
    # Put the shifted characters back
    for i, pos in enumerate(positions):
        S[pos] = shifted_chars[i]

print(''.join(S))