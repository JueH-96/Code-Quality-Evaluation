def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]

    def check_contiguous(cells):
        for i in range(4):
            r1, c1 = cells[i]
            r2, c2 = cells[i+1]
            if abs(r1 - r2) > 1 or abs(c1 - c2) > 1:
                return False
        
        # Check if centers are on a common line at regular intervals
        dx = cells[1][0] - cells[0][0]
        dy = cells[1][1] - cells[0][1]
        
        for i in range(2, 5):
            if cells[i][0] - cells[i-1][0] != dx or cells[i][1] - cells[i-1][1] != dy:
                return False
        
        return True

    for r1 in range(H):
        for c1 in range(W):
            if grid[r1][c1] == 's':
                for r2 in range(H):
                    for c2 in range(W):
                        if grid[r2][c2] == 'n':
                            for r3 in range(H):
                                for c3 in range(W):
                                    if grid[r3][c3] == 'u':
                                        for r4 in range(H):
                                            for c4 in range(W):
                                                if grid[r4][c4] == 'k':
                                                    for r5 in range(H):
                                                        for c5 in range(W):
                                                            if grid[r5][c5] == 'e':
                                                                cells = [(r1, c1), (r2, c2), (r3, c3), (r4, c4), (r5, c5)]
                                                                if check_contiguous(cells):
                                                                    print(r1 + 1, c1 + 1)
                                                                    print(r2 + 1, c2 + 1)
                                                                    print(r3 + 1, c3 + 1)
                                                                    print(r4 + 1, c4 + 1)
                                                                    print(r5 + 1, c5 + 1)
                                                                    return

solve()