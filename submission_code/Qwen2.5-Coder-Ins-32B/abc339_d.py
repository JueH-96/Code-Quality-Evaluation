# YOUR CODE HERE
from collections import deque

def is_valid(x, y, N, grid, visited, player_positions):
    return 0 <= x < N and 0 <= y < N and grid[x][y] != '#' and (x, y) not in visited and (x, y) not in player_positions

def bfs(N, grid, player_positions):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(player_positions, 0)])
    visited = set()
    visited.add(tuple(sorted(player_positions)))
    
    while queue:
        (pos1, pos2), moves = queue.popleft()
        
        if pos1 == pos2:
            return moves
        
        new_positions = set()
        for dx, dy in directions:
            new_pos1 = (pos1[0] + dx, pos1[1] + dy) if is_valid(pos1[0] + dx, pos1[1] + dy, N, grid, visited, {pos2}) else pos1
            new_pos2 = (pos2[0] + dx, pos2[1] + dy) if is_valid(pos2[0] + dx, pos2[1] + dy, N, grid, visited, {new_pos1}) else pos2
            
            if new_pos1 != new_pos2:
                new_positions.add((new_pos1, new_pos2))
                new_positions.add((new_pos2, new_pos1))
        
        for positions in new_positions:
            if tuple(sorted(positions)) not in visited:
                visited.add(tuple(sorted(positions)))
                queue.append((positions, moves + 1))
    
    return -1

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    grid = input[1:]
    
    player_positions = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                player_positions.append((i, j))
    
    result = bfs(N, grid, tuple(player_positions))
    print(result)

if __name__ == "__main__":
    main()