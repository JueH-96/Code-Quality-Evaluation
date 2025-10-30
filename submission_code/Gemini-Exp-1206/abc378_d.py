def solve():
    H, W, K = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] == '.'

    def count_paths(start_r, start_c):
        count = 0
        
        def backtrack(r, c, path):
            nonlocal count
            if len(path) == K + 1:
                count += 1
                return

            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for nr, nc in neighbors:
                if is_valid(nr, nc) and (nr, nc) not in path:
                    backtrack(nr, nc, path + [(nr, nc)])

        backtrack(start_r, start_c, [(start_r, start_c)])
        return count

    total_paths = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                total_paths += count_paths(r, c)

    print(total_paths)

solve()