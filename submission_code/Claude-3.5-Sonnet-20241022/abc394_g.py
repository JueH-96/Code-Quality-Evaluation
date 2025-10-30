from collections import deque
import sys

def read_input():
    H, W = map(int, input().split())
    F = []
    for _ in range(H):
        F.append(list(map(int, input().split())))
    Q = int(input())
    queries = []
    for _ in range(Q):
        a, b, y, c, d, z = map(int, input().split())
        queries.append((a-1, b-1, y, c-1, d-1, z))  # Convert to 0-based indexing
    return H, W, F, Q, queries

def solve_query(H, W, F, start_row, start_col, start_floor, end_row, end_col, end_floor):
    # BFS state: (row, col, floor, stairs_used)
    visited = set()
    queue = deque([(start_row, start_col, start_floor, 0)])
    visited.add((start_row, start_col, start_floor))
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        row, col, floor, stairs = queue.popleft()
        
        # If we reached the target position
        if row == end_row and col == end_col and floor == end_floor:
            return stairs
            
        # Try moving up/down in current building
        for new_floor in [floor-1, floor+1]:
            if 1 <= new_floor <= F[row][col] and (row, col, new_floor) not in visited:
                visited.add((row, col, new_floor))
                queue.append((row, col, new_floor, stairs + 1))
        
        # Try moving to adjacent buildings at same floor
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < H and 0 <= new_col < W and 
                floor <= F[new_row][new_col] and 
                (new_row, new_col, floor) not in visited):
                visited.add((new_row, new_col, floor))
                queue.append((new_row, new_col, floor, stairs))

def main():
    H, W, F, Q, queries = read_input()
    
    # Process each query
    for start_row, start_col, start_floor, end_row, end_col, end_floor in queries:
        result = solve_query(H, W, F, start_row, start_col, start_floor, 
                           end_row, end_col, end_floor)
        print(result)

if __name__ == "__main__":
    main()