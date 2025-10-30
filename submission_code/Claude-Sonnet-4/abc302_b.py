# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

target = "snuke"
directions = [
    (-1, -1), (-1, 0), (-1, 1),  # up-left, up, up-right
    (0, -1),           (0, 1),   # left, right
    (1, -1),  (1, 0),  (1, 1)    # down-left, down, down-right
]

def is_valid_position(r, c):
    return 0 <= r < H and 0 <= c < W

def find_snuke():
    # Try each starting position
    for start_r in range(H):
        for start_c in range(W):
            if grid[start_r][start_c] != 's':
                continue
            
            # Try each direction
            for dr, dc in directions:
                positions = []
                valid = True
                
                # Check if we can form "snuke" in this direction
                for i in range(5):
                    r = start_r + i * dr
                    c = start_c + i * dc
                    
                    if not is_valid_position(r, c):
                        valid = False
                        break
                    
                    if grid[r][c] != target[i]:
                        valid = False
                        break
                    
                    positions.append((r + 1, c + 1))  # Convert to 1-indexed
                
                if valid:
                    return positions
    
    return None

result = find_snuke()
for r, c in result:
    print(r, c)