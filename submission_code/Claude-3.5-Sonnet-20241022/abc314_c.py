N, M = map(int, input().split())
S = list(input())
C = list(map(int, input().split()))

for color in range(1, M+1):
    # Find positions of current color
    positions = [i for i in range(N) if C[i] == color]
    if not positions:
        continue
        
    # Get characters at those positions
    chars = [S[i] for i in positions]
    
    # Perform right circular shift
    chars = [chars[-1]] + chars[:-1]
    
    # Put shifted characters back
    for pos, char in zip(positions, chars):
        S[pos] = char

print(''.join(S))