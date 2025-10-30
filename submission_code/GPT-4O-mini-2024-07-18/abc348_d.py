from collections import deque

def can_reach_goal(H, W, grid, medicines):
    start = None
    goal = None
    
    # Locate the start (S) and goal (T) positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    # Directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS setup
    queue = deque()
    queue.append((start[0], start[1], 0))  # (row, col, current_energy)
    visited = set()
    visited.add((start[0], start[1], 0))
    
    # Store medicines in a dictionary for quick access
    medicine_dict = {}
    for r, c, e in medicines:
        medicine_dict[(r - 1, c - 1)] = e  # Convert to 0-indexed
    
    while queue:
        x, y, energy = queue.popleft()
        
        # Check if we reached the goal
        if (x, y) == goal:
            return "Yes"
        
        # Use medicine if available
        if (x, y) in medicine_dict:
            energy = max(energy, medicine_dict[(x, y)])
        
        # Try to move in all directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                if energy > 0 and (nx, ny, energy - 1) not in visited:
                    visited.add((nx, ny, energy - 1))
                    queue.append((nx, ny, energy - 1))
    
    return "No"

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = [data[i + 1] for i in range(H)]
    N = int(data[H + 1])
    
    medicines = []
    for i in range(N):
        r, c, e = map(int, data[H + 2 + i].split())
        medicines.append((r, c, e))
    
    result = can_reach_goal(H, W, grid, medicines)
    print(result)

if __name__ == "__main__":
    main()