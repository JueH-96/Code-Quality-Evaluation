def main():
    import sys
    input = sys.stdin.read().split()
    
    H = int(input[0])
    W = int(input[1])
    X = int(input[2])
    Y = int(input[3])
    
    grid = []
    index = 4
    for _ in range(H):
        index += 1
        grid.append(list(input[index]))
    
    T = input[index + 1]
    
    # Convert to 0-indexing
    x = X - 1
    y = Y - 1
    
    visited_houses = set()
    
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    for direction in T:
        dx, dy = directions[direction]
        new_x = x + dx
        new_y = y + dy
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            x, y = new_x, new_y
        if grid[x][y] == '@':
            visited_houses.add((x, y))
    
    final_X = x + 1
    final_Y = y + 1
    C = len(visited_houses)
    
    print(final_X, final_Y, C)

if __name__ == "__main__":
    main()