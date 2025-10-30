import heapq

def main():
    # Read input
    H, W = map(int, input().split())
    floors = []
    for _ in range(H):
        floors.append(list(map(int, input().split())))
    
    Q = int(input())
    for _ in range(Q):
        ai, bi, yi, ci, di, zi = map(int, input().split())
        # Convert to 0-indexed
        ai -= 1
        bi -= 1
        ci -= 1
        di -= 1
        
        # Find minimum stairs using Dijkstra's algorithm
        print(min_stairs(H, W, floors, ai, bi, yi, ci, di, zi))

def min_stairs(H, W, floors, start_row, start_col, start_floor, end_row, end_col, end_floor):
    # Priority queue stores (stair_usage, row, col, floor)
    pq = [(0, start_row, start_col, start_floor)]
    
    # Dictionary to keep track of minimum stair usage for each state
    visited = {}
    
    while pq:
        stairs, row, col, floor = heapq.heappop(pq)
        
        # If we've processed this state before with a better cost, skip
        if (row, col, floor) in visited and visited[(row, col, floor)] < stairs:
            continue
        
        # If we've reached the target
        if (row, col, floor) == (end_row, end_col, end_floor):
            return stairs
        
        visited[(row, col, floor)] = stairs
        
        # Try moving up using stairs
        if floor < floors[row][col]:
            new_stairs = stairs + 1
            if (row, col, floor + 1) not in visited or new_stairs < visited[(row, col, floor + 1)]:
                heapq.heappush(pq, (new_stairs, row, col, floor + 1))
        
        # Try moving down using stairs
        if floor > 1:
            new_stairs = stairs + 1
            if (row, col, floor - 1) not in visited or new_stairs < visited[(row, col, floor - 1)]:
                heapq.heappush(pq, (new_stairs, row, col, floor - 1))
        
        # Try using walkways to adjacent blocks
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and floors[new_row][new_col] >= floor:
                if (new_row, new_col, floor) not in visited or stairs < visited[(new_row, new_col, floor)]:
                    heapq.heappush(pq, (stairs, new_row, new_col, floor))
    
    # If no path is found
    return -1  # This should not happen given the problem statement

if __name__ == "__main__":
    main()