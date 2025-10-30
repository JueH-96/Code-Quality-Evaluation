import heapq

def main():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input()))
    
    # Find start (S) and goal (T) positions
    si = sj = ti = tj = -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                si, sj = i, j
            elif grid[i][j] == 'T':
                ti, tj = i, j
    
    N = int(input())
    medicine = [[0] * W for _ in range(H)]
    for _ in range(N):
        r, c, e = map(int, input().split())
        medicine[r-1][c-1] = e
    
    max_energy = [[-1 for _ in range(W)] for _ in range(H)]
    heap = []
    max_energy[si][sj] = 0
    heapq.heappush(heap, (0, si, sj))  # Stored as negative for max-heap simulation
    
    found = False
    while heap:
        current_e_neg, i, j = heapq.heappop(heap)
        current_e = -current_e_neg
        
        # Check if current state is outdated
        if max_energy[i][j] > current_e:
            continue
        
        # Check if goal is reached
        if i == ti and j == tj:
            found = True
            break
        
        # Process medicine at current cell
        if medicine[i][j] > 0:
            e_med = medicine[i][j]
            if max_energy[i][j] < e_med:
                max_energy[i][j] = e_med
                heapq.heappush(heap, (-e_med, i, j))
                medicine[i][j] = 0  # Mark medicine as used
        
        # Process movement if current energy is positive
        if current_e > 0:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                ni = i + dx
                nj = j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] != '#':
                        new_e = current_e - 1
                        if new_e > max_energy[ni][nj]:
                            max_energy[ni][nj] = new_e
                            heapq.heappush(heap, (-new_e, ni, nj))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()