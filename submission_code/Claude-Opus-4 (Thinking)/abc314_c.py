# YOUR CODE HERE
N, M = map(int, input().split())
S = input()
C = list(map(int, input().split()))

# Convert string to list for easier manipulation
S_list = list(S)

# Process each color from 1 to M
for color in range(1, M + 1):
    # Find all positions with the current color
    positions = []
    for i in range(N):
        if C[i] == color:
            positions.append(i)
    
    # If there's only one character with this color, no shift needed
    if len(positions) <= 1:
        continue
    
    # Extract characters at these positions
    chars = [S_list[pos] for pos in positions]
    
    # Perform right circular shift by 1
    # The last character goes to the front
    shifted_chars = [chars[-1]] + chars[:-1]
    
    # Place the shifted characters back
    for i, pos in enumerate(positions):
        S_list[pos] = shifted_chars[i]

# Convert back to string and print
print(''.join(S_list))