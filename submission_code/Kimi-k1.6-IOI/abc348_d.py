import heapq

def main():
    H, W = map(int, input().split())
    grid = []
    start = None
    goal = None
    for i in range(H):
        line = input().strip()
        grid.append(line)
        for j in range(W):
            if line[j] == 'S':
                start = (i, j)
            elif line[j] == 'T':
                goal = (i, j)
    
    N = int(input())
    medicine_e = [[0]*W for _ in range(H)]
    for _ in range(N):
        R, C, E = map(int, input().split())
        R0 = R - 1
        C0 = C - 1
        medicine_e[R0][C0] = E
    
    max_energy = [[-1 for _ in range(W)] for __ in range(H)]
    i_start, j_start = start
    max_energy[i_start][j_start] = 0
    heap = []
    heapq.heappush(heap, (0, i_start, j_start))  # Stored as negative energy
    
    found = False
    while heap:
        e_stored, i, j = heapq.heappop(heap)
        current_e = -e_stored
        
        if current_e < max_energy[i][j]:
            continue
        
        if (i, j) == goal:
            found = True
            break
        
        # Explore all four directions
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#':
                new_e = current_e - 1
                if new_e >= 0 and new_e > max_energy[ni][nj]:
                    max_energy[ni][nj] = new_e
                    heapq.heappush(heap, (-new_e, ni, nj))
        
        # Check medicine usage
        if medicine_e[i][j] > 0:
            E_m = medicine_e[i][j]
            if E_m > current_e and E_m > max_energy[i][j]:
                max_energy[i][j] = E_m
                heapq.heappush(heap, (-E_m, i, j))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()