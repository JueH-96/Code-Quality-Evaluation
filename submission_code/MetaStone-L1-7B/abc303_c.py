# Read input
N, M, H, K = map(int, input().split())
S = input().strip()
items = [tuple(map(int, input().split())) for _ in range(M)]

positions = set()
x, y = 0, 0
h = H

for c in S:
    # Update position based on the character
    if c == 'R':
        x += 1
    elif c == 'L':
        x -= 1
    elif c == 'U':
        y += 1
    elif c == 'D':
        y -= 1
    
    # Deduct 1 from health
    h -= 1
    
    # Check if health is negative
    if h < 0:
        print("No")
        exit()
    
    # Check if the new position has an item and if health allows taking it
    if (x, y) in positions:
        continue
    if h < K:
        h = K
        positions.add((x, y))

# All moves completed successfully
print("Yes")