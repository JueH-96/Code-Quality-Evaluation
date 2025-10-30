N, M, H, K = map(int, input().split())
S = input().strip()

# Store item locations
items = set()
for _ in range(M):
    x, y = map(int, input().split())
    items.add((x, y))

# Start position and health
x, y = 0, 0
health = H

# Process each move
result = "Yes"
for i in range(N):
    # Move based on direction
    if S[i] == 'R':
        x += 1
    elif S[i] == 'L':
        x -= 1
    elif S[i] == 'U':
        y += 1
    elif S[i] == 'D':
        y -= 1
    
    # Consume health for the move
    health -= 1
    
    # Check if collapsed
    if health < 0:
        result = "No"
        break
    
    # Check if there's an item and health < K
    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))  # Item is consumed

print(result)