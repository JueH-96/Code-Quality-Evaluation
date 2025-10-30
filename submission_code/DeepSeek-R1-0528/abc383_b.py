def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W, D = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + H):
        grid.append(data[i].strip())
    
    floor_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
                
    n = len(floor_cells)
    max_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            A = floor_cells[i]
            B = floor_cells[j]
            count = 0
            for cell in floor_cells:
                d1 = abs(cell[0] - A[0]) + abs(cell[1] - A[1])
                d2 = abs(cell[0] - B[0]) + abs(cell[1] - B[1])
                if d1 <= D or d2 <= D:
                    count += 1
            if count > max_count:
                max_count = count
                
    print(max_count)

if __name__ == "__main__":
    main()