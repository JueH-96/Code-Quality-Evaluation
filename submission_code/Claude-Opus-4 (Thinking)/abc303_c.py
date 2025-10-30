N, M, H, K = map(int, input().split())
S = input().strip()

# Store items in a set for O(1) lookup
items = set()
for _ in range(M):
    x, y = map(int, input().split())
    items.add((x, y))

# Initial position and health
x, y = 0, 0
health = H

# Process each move
for move in S:
    # Move based on direction
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    
    # Consume health
    health -= 1
    
    # Check if collapsed
    if health < 0:
        print("No")
        exit()
    
    # Check if item is at current position and health < K
    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))  # Item is consumed

print("Yes")