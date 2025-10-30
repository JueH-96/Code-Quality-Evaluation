def solve():
    n = int(input())
    s = [input() for _ in range(n)]
    grid = []
    for i in range(n):
        row = []
        for char in s[i]:
            row.append(char)
        grid.append(row)
    
    count = 0
    for r1 in range(n):
        for r2 in range(r1 + 1, n):
            for c1 in range(n):
                for c2 in range(c1 + 1, n):
                    # Triple 1: (r1, c1), (r1, c2), (r2, c1)
                    if grid[r1][c1] == 'o' and grid[r1][c2] == 'o' and grid[r2][c1] == 'o':
                        count += 1
                    # Triple 2: (r1, c1), (r1, c2), (r2, c2)
                    if grid[r1][c1] == 'o' and grid[r1][c2] == 'o' and grid[r2][c2] == 'o':
                        count += 1
                    # Triple 3: (r1, c1), (r2, c1), (r2, c2)
                    if grid[r1][c1] == 'o' and grid[r2][c1] == 'o' and grid[r2][c2] == 'o':
                        count += 1
                    # Triple 4: (r1, c2), (r2, c1), (r2, c2)
                    if grid[r1][c2] == 'o' and grid[r2][c1] == 'o' and grid[r2][c2] == 'o':
                        count += 1
                        
    print(count)

if __name__ == '__main__':
    solve()