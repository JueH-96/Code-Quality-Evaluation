def main():
    import sys
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    grid = []
    for i in range(1, 1+n):
        grid.append(data[i].strip())
    
    answers = []
    for i in range(n-8):
        for j in range(m-8):
            valid = True
            
            for r in range(i, i+3):
                for c in range(j, j+3):
                    if grid[r][c] != '#':
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            for r in range(i+6, i+9):
                for c in range(j+6, j+9):
                    if grid[r][c] != '#':
                        valid = False
                        break
                if not valid:
                    break
                    
            if not valid:
                continue
                
            adj_top = [
                (i, j+3), (i+1, j+3), (i+2, j+3),
                (i+3, j), (i+3, j+1), (i+3, j+2),
                (i+3, j+3)
            ]
            for (r, c) in adj_top:
                if grid[r][c] != '.':
                    valid = False
                    break
                    
            if not valid:
                continue
                
            adj_bottom = [
                (i+5, j+6), (i+5, j+7), (i+5, j+8),
                (i+6, j+5), (i+7, j+5), (i+8, j+5),
                (i+5, j+5)
            ]
            for (r, c) in adj_bottom:
                if grid[r][c] != '.':
                    valid = False
                    break
                    
            if valid:
                answers.append((i, j))
                
    for (i, j) in answers:
        print(f"{i+1} {j+1}")

if __name__ == "__main__":
    main()