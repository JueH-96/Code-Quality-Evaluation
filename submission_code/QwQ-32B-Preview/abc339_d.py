from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = [data[i+1] for i in range(N)]
    
    # Find initial positions of the players
    pos1, pos2 = None, None
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                if pos1 is None:
                    pos1 = (i, j)
                else:
                    pos2 = (i, j)
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Function to move a player in a given direction
    def move(pos, dr, dc):
        new_row = pos[0] + dr
        new_col = pos[1] + dc
        if 0 <= new_row < N and 0 <= new_col < N and grid[new_row][new_col] != '#':
            return (new_row, new_col)
        else:
            return pos
    
    # BFS
    queue = deque()
    visited = set()
    initial_state = (pos1, pos2)
    queue.append((initial_state, 0))
    visited.add(initial_state)
    
    while queue:
        current, moves = queue.popleft()
        pos1, pos2 = current
        if pos1 == pos2:
            print(moves)
            return
        for dr, dc in directions:
            new_pos1 = move(pos1, dr, dc)
            new_pos2 = move(pos2, dr, dc)
            new_state = (new_pos1, new_pos2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + 1))
    print(-1)

if __name__ == "__main__":
    main()