import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    X = int(input[idx]); idx +=1
    P = int(input[idx]) -1; idx +=1
    Q = int(input[idx]) -1; idx +=1
    
    grid = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        idx += W
        grid.append(row)
    
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    current = grid[P][Q]
    absorbed = [[False]*W for _ in range(H)]
    in_queue = [[False]*W for _ in range(H)]
    absorbed[P][Q] = True
    
    heap = []
    
    for dx, dy in directions:
        ni = P + dx
        nj = Q + dy
        if 0 <= ni < H and 0 <= nj < W:
            if not in_queue[ni][nj]:
                in_queue[ni][nj] = True
                s = grid[ni][nj]
                x = s * X
                heapq.heappush(heap, (x, s, ni, nj))
    
    while heap:
        x, s, i, j = heapq.heappop(heap)
        if absorbed[i][j]:
            continue
        if current > x:
            current += s
            absorbed[i][j] = True
            for dx, dy in directions:
                ni = i + dx
                nj = j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    if not absorbed[ni][nj] and not in_queue[ni][nj]:
                        in_queue[ni][nj] = True
                        s_new = grid[ni][nj]
                        x_new = s_new * X
                        heapq.heappush(heap, (x_new, s_new, ni, nj))
        else:
            break
    
    print(current)

if __name__ == "__main__":
    main()