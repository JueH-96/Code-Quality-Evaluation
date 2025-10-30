import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    grid = []
    for _ in range(H):
        grid.append(data[idx].strip())
        idx += 1
    
    N = int(data[idx])
    idx += 1
    
    medicines = []
    for _ in range(N):
        R = int(data[idx]) - 1
        idx += 1
        C = int(data[idx]) - 1
        idx += 1
        E = int(data[idx])
        idx += 1
        medicines.append((R, C, E))
    
    # Find start and goal positions
    si = sj = ti = tj = -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                si, sj = i, j
            if grid[i][j] == 'T':
                ti, tj = i, j
    
    # Initialize medicine_energy grid
    medicine_energy = [[None] * W for _ in range(H)]
    for r, c, e in medicines:
        medicine_energy[r][c] = e
    
    # Initialize max_energy grid with -1 (minimum possible)
    max_energy = [[-1] * W for _ in range(H)]
    max_energy[si][sj] = 0  # Start with 0 energy
    
    heap = []
    heapq.heappush(heap, (0, si, sj))  # Push with key -e (initial e=0, key is 0)
    
    found = False
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while heap:
        key, i, j = heapq.heappop(heap)
        current_e = -key
        
        if i == ti and j == tj:
            found = True
            break
        
        if current_e < max_energy[i][j]:
            continue
        
        # Explore four directions
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#':
                new_e = current_e - 1
                if new_e >= 0 and max_energy[ni][nj] < new_e:
                    max_energy[ni][nj] = new_e
                    heapq.heappush(heap, (-new_e, ni, nj))
        
        # Check medicine at current cell
        if medicine_energy[i][j] is not None:
            med_e = medicine_energy[i][j]
            if med_e > max_energy[i][j]:
                max_energy[i][j] = med_e
                heapq.heappush(heap, (-med_e, i, j))
                medicine_energy[i][j] = None  # Mark as used
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()