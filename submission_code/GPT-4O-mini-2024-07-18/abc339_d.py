from collections import deque

def min_moves_to_meet(N, grid):
    # Find the positions of the players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    # Directions for moving: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS to find the minimum moves
    def bfs(start1, start2):
        queue = deque([(start1[0], start1[1], start2[0], start2[1], 0)])  # (x1, y1, x2, y2, moves)
        visited = set()
        visited.add((start1[0], start1[1], start2[0], start2[1]))
        
        while queue:
            x1, y1, x2, y2, moves = queue.popleft()
            
            # Check if they are at the same position
            if (x1, y1) == (x2, y2):
                return moves
            
            # Try all possible moves
            for dx1, dy1 in directions:
                nx1, ny1 = x1 + dx1, y1 + dy1
                if 0 <= nx1 < N and 0 <= ny1 < N and grid[nx1][ny1] != '#':
                    new_pos1 = (nx1, ny1)
                else:
                    new_pos1 = (x1, y1)  # Player 1 doesn't move
                
                for dx2, dy2 in directions:
                    nx2, ny2 = x2 + dx2, y2 + dy2
                    if 0 <= nx2 < N and 0 <= ny2 < N and grid[nx2][ny2] != '#':
                        new_pos2 = (nx2, ny2)
                    else:
                        new_pos2 = (x2, y2)  # Player 2 doesn't move
                    
                    if (new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1]) not in visited:
                        visited.add((new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1]))
                        queue.append((new_pos1[0], new_pos1[1], new_pos2[0], new_pos2[1], moves + 1))
        
        return -1  # If they cannot meet
    
    return bfs(players[0], players[1])

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
N = int(data[0])
grid = data[1:N+1]

# Get the result and print it
result = min_moves_to_meet(N, grid)
print(result)