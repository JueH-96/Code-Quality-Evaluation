from collections import deque

def solve_query(H, W, F, start_row, start_col, start_floor, end_row, end_col, end_floor):
    # 0-1 BFS
    dq = deque()
    dist = {}
    
    # Convert to 0-indexed
    start_row -= 1
    start_col -= 1
    end_row -= 1
    end_col -= 1
    
    start_state = (start_row, start_col, start_floor)
    end_state = (end_row, end_col, end_floor)
    
    dq.append(start_state)
    dist[start_state] = 0
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while dq:
        row, col, floor = dq.popleft()
        
        if (row, col, floor) == end_state:
            return dist[end_state]
        
        current_dist = dist[(row, col, floor)]
        
        # Try moving to adjacent buildings on the same floor (0 cost)
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W:
                if F[new_row][new_col] >= floor:
                    new_state = (new_row, new_col, floor)
                    if new_state not in dist:
                        dist[new_state] = current_dist
                        dq.appendleft(new_state)
        
        # Try moving up one floor in the same building (1 cost)
        if floor < F[row][col]:
            new_state = (row, col, floor + 1)
            if new_state not in dist or dist[new_state] > current_dist + 1:
                dist[new_state] = current_dist + 1
                dq.append(new_state)
        
        # Try moving down one floor in the same building (1 cost)
        if floor > 1:
            new_state = (row, col, floor - 1)
            if new_state not in dist or dist[new_state] > current_dist + 1:
                dist[new_state] = current_dist + 1
                dq.append(new_state)
    
    return -1  # Should not reach here if input is valid

# Read input
H, W = map(int, input().split())
F = []
for _ in range(H):
    F.append(list(map(int, input().split())))

Q = int(input())
for _ in range(Q):
    A, B, Y, C, D, Z = map(int, input().split())
    result = solve_query(H, W, F, A, B, Y, C, D, Z)
    print(result)