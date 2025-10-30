def solve():
    N, M, H, K = map(int, input().split())
    S = input().strip()
    
    # Store positions of items in a set for O(1) lookup
    items = set()
    for _ in range(M):
        x, y = map(int, input().split())
        items.add((x, y))
    
    # Initialize Takahashi's position and health
    x, y = 0, 0
    health = H
    
    # Dictionary to map moves to direction
    directions = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }
    
    for move in S:
        # Update position based on the move
        dx, dy = directions[move]
        x += dx
        y += dy
        
        # Decrease health by 1 for each move
        health -= 1
        
        # Check if Takahashi has collapsed
        if health < 0:
            return "No"
        
        # Check if there's an item at current position and health is low
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))  # Remove the item after consumption
    
    return "Yes"

print(solve())