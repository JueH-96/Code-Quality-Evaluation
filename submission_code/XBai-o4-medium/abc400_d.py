import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    
    grid = []
    for _ in range(H):
        line = input[idx]
        idx +=1
        grid.append(list(line))
    
    A = int(input[idx]); idx +=1
    B = int(input[idx]); idx +=1
    C = int(input[idx]); idx +=1
    D = int(input[idx]); idx +=1
    
    start_i, start_j = A-1, B-1
    end_i, end_j = C-1, D-1
    
    INF = float('inf')
    kicks_needed = [[INF for _ in range(W)] for _ in range(H)]
    kicks_needed[start_i][start_j] = 0
    
    heap = []
    heapq.heappush(heap, (0, start_i, start_j))
    
    dirs = [ (-1, 0), (1, 0), (0, -1), (0, 1) ]
    
    while heap:
        current_k, i, j = heapq.heappop(heap)
        if i == end_i and j == end_j:
            print(current_k)
            return
        if current_k > kicks_needed[i][j]:
            continue
        
        # Process moving to adjacent cells
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.':
                    if kicks_needed[ni][nj] > current_k:
                        kicks_needed[ni][nj] = current_k
                        heapq.heappush(heap, (current_k, ni, nj))
        
        # Process front kicks in each direction
        for kick_dir in dirs:
            kx, ky = kick_dir
            # Perform the front kick, modify grid
            for step in [1, 2]:
                nx = i + kx * step
                ny = j + ky * step
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] == '#':
                        grid[nx][ny] = '.'
            # Check adjacent cells after modification
            for dx, dy in dirs:
                ni = i + dx
                nj = j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] == '.':
                        if kicks_needed[ni][nj] > current_k + 1:
                            kicks_needed[ni][nj] = current_k + 1
                            heapq.heappush(heap, (current_k + 1, ni, nj))
    
    # If not found (should not happen as per problem statement)
    print(-1)

if __name__ == "__main__":
    main()