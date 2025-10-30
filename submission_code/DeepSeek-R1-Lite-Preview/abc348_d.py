import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W = map(int, data[0].split())
    grid = [list(data[i+1]) for i in range(H)]
    
    S = T = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S = (i, j)
            if grid[i][j] == 'T':
                T = (i, j)
    
    N = int(data[H+1])
    medicines = []
    for i in range(N):
        R, C, E = map(int, data[H+2+i].split())
        r = R - 1
        c = C - 1
        medicines.append((r, c, E))
    
    medicine_dict = {}
    for med in medicines:
        medicine_dict[(med[0], med[1])] = med[2]
    
    if S in medicine_dict:
        start_medicines = [(S[0], S[1], medicine_dict[S])]
    else:
        print("No")
        return
    
    heap = []
    for med in start_medicines:
        E_i = med[2]
        heapq.heappush(heap, (-E_i, med[0], med[1]))
    
    visited = [[-1 for _ in range(W)] for _ in range(H)]
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while heap:
        neg_energy, row, col = heapq.heappop(heap)
        energy = -neg_energy
        if (row, col) == T:
            print("Yes")
            return
        if energy <= visited[row][col]:
            continue
        visited[row][col] = energy
        for d in dirs:
            new_row = row + d[0]
            new_col = col + d[1]
            if 0 <= new_row < H and 0 <= new_col < W and grid[new_row][new_col] != '#':
                new_energy = energy - 1
                if new_energy >= 0 and new_energy > visited[new_row][new_col]:
                    visited[new_row][new_col] = new_energy
                    heapq.heappush(heap, (-new_energy, new_row, new_col))
        if (row, col) in medicine_dict:
            E_i = medicine_dict[(row, col)]
            if E_i > visited[row][col]:
                visited[row][col] = E_i
                heapq.heappush(heap, (-E_i, row, col))
    
    print("No")

if __name__ == "__main__":
    main()