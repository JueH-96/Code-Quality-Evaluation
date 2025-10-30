# YOUR CODE HERE
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
    # Move based on direction
    if S[i] == 'R':
        x += 1
    elif S[i] == 'L':
        x -= 1
    elif S[i] == 'U':
        y += 1
    elif S[i] == 'D':
        y -= 1
    
    # Consume 1 health for the move
    health -= 1
    
    # Check if collapsed
    if health < 0:
        print("No")
        exit()
    
    # Check if there's an item at current position and health < K
    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))  # Remove consumed item

# If we completed all moves without collapsing
print("Yes")