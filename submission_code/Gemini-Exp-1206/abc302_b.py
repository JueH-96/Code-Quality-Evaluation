def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def check_contiguous(cells):
        for i in range(4):
            r1, c1 = cells[i]
            r2, c2 = cells[i+1]
            if not (abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1):
                return False
        return True

    def check_line(cells):
        if len(cells) != 5:
            return False
        
        r_diffs = [cells[i+1][0] - cells[i][0] for i in range(4)]
        c_diffs = [cells[i+1][1] - cells[i][1] for i in range(4)]
        
        if all(x == r_diffs[0] for x in r_diffs) and all(x == c_diffs[0] for x in c_diffs):
            return True
        else:
            return False

    def find_snuke():
        for r1 in range(h):
            for c1 in range(w):
                if grid[r1][c1] == 's':
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0:
                                continue
                            
                            cells = [(r1, c1)]
                            r, c = r1, c1
                            valid = True
                            for i in range(1, 5):
                                r += dr
                                c += dc
                                if 0 <= r < h and 0 <= c < w:
                                    if "snuke"[i] == grid[r][c]:
                                        cells.append((r, c))
                                    else:
                                        valid = False
                                        break
                                else:
                                    valid = False
                                    break
                            
                            if valid and check_contiguous(cells) and check_line(cells):
                                return cells
        return None

    result = find_snuke()
    for r, c in result:
        print(r + 1, c + 1)

solve()