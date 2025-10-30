def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, m = map(int, data[0].split())
    grid = []
    for i in range(1, 1 + n):
        grid.append(data[i].strip())
    
    results = []
    top_left_black = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2)]
    
    top_left_adjacent = [(0, 3), (1, 3), (2, 3),
                         (3, 0), (3, 1), (3, 2),
                         (3, 3)]
    
    bottom_right_black = [(6, 6), (6, 7), (6, 8),
                          (7, 6), (7, 7), (7, 8),
                          (8, 6), (8, 7), (8, 8)]
    
    bottom_right_adjacent = [(5, 5), (5, 6), (5, 7), (5, 8),
                             (6, 5), (7, 5), (8, 5)]
    
    for i in range(0, n - 8):
        for j in range(0, m - 8):
            valid = True
            for dr, dc in top_left_black:
                r = i + dr
                c = j + dc
                if grid[r][c] != '#':
                    valid = False
                    break
            if not valid:
                continue
                
            for dr, dc in top_left_adjacent:
                r = i + dr
                c = j + dc
                if grid[r][c] != '.':
                    valid = False
                    break
            if not valid:
                continue
                
            for dr, dc in bottom_right_black:
                r = i + dr
                c = j + dc
                if grid[r][c] != '#':
                    valid = False
                    break
            if not valid:
                continue
                
            for dr, dc in bottom_right_adjacent:
                r = i + dr
                c = j + dc
                if grid[r][c] != '.':
                    valid = False
                    break
                    
            if valid:
                results.append((i+1, j+1))
                
    results.sort()
    for res in results:
        print(f"{res[0]} {res[1]}")

if __name__ == "__main__":
    main()