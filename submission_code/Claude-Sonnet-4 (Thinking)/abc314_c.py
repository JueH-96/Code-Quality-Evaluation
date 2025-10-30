n, m = map(int, input().split())
s = list(input())
colors = list(map(int, input().split()))

for color in range(1, m + 1):
    # Find positions with this color
    positions = []
    for i in range(n):
        if colors[i] == color:
            positions.append(i)
    
    # Get characters at these positions
    chars = [s[pos] for pos in positions]
    
    # Perform right circular shift by 1
    if len(chars) > 1:
        shifted_chars = [chars[-1]] + chars[:-1]
    else:
        shifted_chars = chars
    
    # Update the string
    for i, pos in enumerate(positions):
        s[pos] = shifted_chars[i]

print(''.join(s))