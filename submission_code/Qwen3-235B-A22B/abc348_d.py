import heapq

def main():
    h, w = map(int, input().split())
    grid = []
    start = None
    goal = None
    for i in range(h):
        line = input().strip()
        grid.append(list(line))
        for j in range(w):
            if grid[i][j] == 'S':
                start = (i+1, j+1)
            elif grid[i][j] == 'T':
                goal = (i+1, j+1)
    
    n = int(input())
    medicine_map = {}
    for _ in range(n):
        r, c, e = map(int, input().split())
        medicine_map[(r, c)] = e
    
    INF = float('-inf')
    max_energy = [[INF] * (w + 2) for _ in range(h + 2)]
    heap = []
    x_s, y_s = start
    x_g, y_g = goal
    heapq.heappush(heap, (0, x_s, y_s))
    max_energy[x_s][y_s] = 0  # stored as negative in heap, but initial is 0
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    found = False
    
    while heap:
        current_energy_neg, x, y = heapq.heappop(heap)
        current_energy = -current_energy_neg
        
        if (x, y) == (x_g, y_g):
            found = True
            break
        
        if current_energy < max_energy[x][y]:
            continue
        
        if (x, y) in medicine_map:
            med_e = medicine_map[(x, y)]
            if med_e > max_energy[x][y]:
                max_energy[x][y] = med_e
                heapq.heappush(heap, (-med_e, x, y))
        
        for dx, dy in dirs:
            nx = x + dx
            ny = y + dy
            if 1 <= nx <= h and 1 <= ny <= w:
                cell_char = grid[nx-1][ny-1]
                if cell_char == '#':
                    continue
                new_energy = current_energy - 1
                if new_energy >= 0 and new_energy > max_energy[nx][ny]:
                    max_energy[nx][ny] = new_energy
                    heapq.heappush(heap, (-new_energy, nx, ny))
    
    print("Yes" if found else "No")

if __name__ == "__main__":
    main()