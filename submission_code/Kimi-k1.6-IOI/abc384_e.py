import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx +=1
    W = int(input[idx])
    idx +=1
    X = int(input[idx])
    idx +=1
    P = int(input[idx])-1  # convert to 0-based
    idx +=1
    Q = int(input[idx])-1  # convert to 0-based
    idx +=1
    
    grid = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        grid.append(row)
        idx += W
    
    sum_total = grid[P][Q]
    absorbed = [[False]*W for _ in range(H)]
    absorbed[P][Q] = True
    
    heap = []
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    for dx, dy in directions:
        nx = P + dx
        ny = Q + dy
        if 0 <= nx < H and 0 <= ny < W and not absorbed[nx][ny]:
            required = grid[nx][ny] * X
            heapq.heappush(heap, (required, nx, ny))
    
    while heap:
        required, x, y = heapq.heappop(heap)
        if absorbed[x][y]:
            continue
        if sum_total > required:
            sum_total += grid[x][y]
            absorbed[x][y] = True
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < H and 0 <= ny < W and not absorbed[nx][ny]:
                    new_required = grid[nx][ny] * X
                    heapq.heappush(heap, (new_required, nx, ny))
        else:
            break
    
    print(sum_total)

if __name__ == "__main__":
    main()