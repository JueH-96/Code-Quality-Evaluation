def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    min_ops = float('inf')

    for r in range(h):
        for c in range(w - k + 1):
            ops = 0
            valid = True
            for i in range(k):
                if grid[r][c+i] == 'x':
                    valid = False
                    break
                elif grid[r][c+i] == '.':
                    ops += 1
            if valid:
                min_ops = min(min_ops, ops)
    
    for r in range(h - k + 1):
        for c in range(w):
            ops = 0
            valid = True
            for i in range(k):
                if grid[r+i][c] == 'x':
                    valid = False
                    break
                elif grid[r+i][c] == '.':
                    ops += 1
            if valid:
                min_ops = min(min_ops, ops)

    if min_ops == float('inf'):
        print("-1")
    else:
        print(min_ops)

solve()