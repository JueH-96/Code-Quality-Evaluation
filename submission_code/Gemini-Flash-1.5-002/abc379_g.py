MOD = 998244353

def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    
    q_count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '?':
                q_count += 1

    count = 0
    for i in range(3**q_count):
        temp_grid = [row[:] for row in grid]
        temp = i
        for r in range(h):
            for c in range(w):
                if temp_grid[r][c] == '?':
                    digit = (temp % 3) + 1
                    temp_grid[r][c] = str(digit)
                    temp //= 3

        valid = True
        for r in range(h):
            for c in range(w):
                if r > 0 and temp_grid[r][c] == temp_grid[r-1][c]:
                    valid = False
                    break
                if c > 0 and temp_grid[r][c] == temp_grid[r][c-1]:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            count += 1

    print(count % MOD)

solve()