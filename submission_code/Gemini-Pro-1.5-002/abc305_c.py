# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    empty_count = 0
    empty_r = -1
    empty_c = -1
    
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                empty_count += 1
                empty_r = r
                empty_c = c
    
    if empty_count != 1:
        print("Error: Invalid input")
        return

    min_r = -1
    max_r = -1
    min_c = -1
    max_c = -1

    for r1 in range(h - 1):
        for c1 in range(w - 1):
            for r2 in range(r1 + 1, h):
                for c2 in range(c1 + 1, w):
                    temp_grid = [['' for _ in range(w)] for _ in range(h)]
                    for r in range(h):
                        for c in range(w):
                            if r1 <= r <= r2 and c1 <= c <= c2:
                                temp_grid[r][c] = '#'
                            else:
                                temp_grid[r][c] = '.'

                    diff_count = 0
                    diff_r = -1
                    diff_c = -1

                    for r in range(h):
                        for c in range(w):
                            if temp_grid[r][c] != grid[r][c]:
                                diff_count += 1
                                diff_r = r
                                diff_c = c

                    if diff_count == 1:
                        print(diff_r + 1, diff_c + 1)
                        return

solve()