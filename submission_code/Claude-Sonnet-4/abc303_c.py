N, M, H, K = map(int, input().split())
S = input().strip()

# Store item positions in a set for O(1) lookup
items = set()
for _ in range(M):
    x, y = map(int, input().split())
    items.add((x, y))

# Starting position and health
x, y = 0, 0
health = H

# Process each move
for i in range(N):
    # Consume 1 health for the move
    health -= 1
    
    # If health becomes negative, collapse
    if health < 0:
        print("No")
        exit()
    
    # Make the move based on direction
    direction = S[i]
    if direction == 'R':
        x += 1
    elif direction == 'L':
        x -= 1
    elif direction == 'U':
        y += 1
    elif direction == 'D':
        y -= 1
    
    # Check if there's an item at current position and health < K
    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))  # Remove the consumed item

# If we completed all moves without collapsing
print("Yes")