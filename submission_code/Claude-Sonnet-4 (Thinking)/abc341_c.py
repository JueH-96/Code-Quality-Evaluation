H, W, N = map(int, input().split())
T = input().strip()
grid = []
for _ in range(H):
    grid.append(input().strip())

directions = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

possible_ends = set()

for start_i in range(H):
    for start_j in range(W):
        if grid[start_i][start_j] == '#':
            continue
        
        current_i, current_j = start_i, start_j
        valid_path = True
        
        for move in T:
            di, dj = directions[move]
            new_i = current_i + di
            new_j = current_j + dj
            
            if new_i < 0 or new_i >= H or new_j < 0 or new_j >= W or grid[new_i][new_j] == '#':
                valid_path = False
                break
            
            current_i, current_j = new_i, new_j
        
        if valid_path:
            possible_ends.add((current_i, current_j))

print(len(possible_ends))