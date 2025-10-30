# Read inputs
N, M = map(int, input().split())
S = list(input().strip())
C = list(map(int, input().split()))

for color in range(1, M+1):
    # Find positions of characters with the current color
    positions = [i for i in range(N) if C[i] == color]
    
    # Store the characters at each position
    chars = [S[i] for i in positions]
    
    # Perform right circular shift by 1
    chars = [chars[-1]] + chars[:-1]
    
    # Update the string with shifted characters
    for i, pos in enumerate(positions):
        S[pos] = chars[i]

print(''.join(S))