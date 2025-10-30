import heapq

def main():
    import sys
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    s_i, s_j = -1, -1
    t_i, t_j = -1, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                s_i, s_j = i, j
            elif grid[i][j] == 'T':
                t_i, t_j = i, j
    
    N = int(sys.stdin.readline())
    med = {}
    for _ in range(N):
        R, C, E = map(int, sys.stdin.readline().split())
        r = R - 1
        c = C - 1
        med[(r, c)] = E
    
    INF = float('-inf')
    max_energy = [[INF for _ in range(W)] for _ in range(H)]
    max_energy[s_i][s_j] = 0
    heap = []
    heapq.heappush(heap, (0, s_i, s_j))  # stored as (-energy, i, j) but using negative for max-heap
    
    found = False
    while heap:
        current_priority, i, j = heapq.heappop(heap)
        current_e = -current_priority
        
        if i == t_i and j == t_j:
            print("Yes")
            return
        
        if current_e < max_energy[i][j]:
            continue
        
        # Check medicine
        if (i, j) in med:
            e_med = med[(i, j)]
            new_e = e_med
            if new_e > max_energy[i][j]:
                max_energy[i][j] = new_e
                heapq.heappush(heap, (-new_e, i, j))
                del med[(i, j)]  # Remove the medicine after taking it
        
        # Move to adjacent cells
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    continue
                next_e = current_e - 1
                if next_e >= 0:
                    if next_e > max_energy[ni][nj]:
                        max_energy[ni][nj] = next_e
                        heapq.heappush(heap, (-next_e, ni, nj))
    
    print("No")

if __name__ == "__main__":
    main()