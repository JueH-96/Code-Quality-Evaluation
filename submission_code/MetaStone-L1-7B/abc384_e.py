import heapq

def main():
    H, W, X = map(int, input().split())
    P = int(input())
    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    
    max_strength = 0
    start = (P-1, P-1)
    current_strength = grid[start[0]][start[1]]
    
    max_heap = []
    heapq.heappush(max_heap, (-current_strength, start[0], start[1]))
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    while max_heap:
        current_neg, i, j = heapq.heappop(max_heap)
        current_strength = -current_neg
        # If this cell's strength is not the current maximum for (i,j), skip
        if current_strength < max_strength:
            continue
        for dx, dy in directions:
            x = i + dx
            y = j + dy
            if 0 <= x < H and 0 <= y < W:
                if grid[x][y] < current_strength:
                    new_strength = current_strength + grid[x][y]
                    if new_strength > max_strength:
                        max_strength = new_strength
                        heapq.heappush(max_heap, (-new_strength, x, y))
    
    print(max_strength)

if __name__ == "__main__":
    main()