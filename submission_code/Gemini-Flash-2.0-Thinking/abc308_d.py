def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    snuke = "snuke"

    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w

    memo = {}

    def dfs(r, c, snuke_index):
        if (r, c, snuke_index) in memo:
            return memo[(r, c, snuke_index)]

        expected_char = snuke[snuke_index % 5]
        if grid[r][c] != expected_char:
            memo[(r, c, snuke_index)] = False
            return False

        if r == h - 1 and c == w - 1:
            memo[(r, c, snuke_index)] = True
            return True

        # Explore adjacent cells
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc):
                if dfs(nr, nc, snuke_index + 1):
                    memo[(r, c, snuke_index)] = True
                    return True

        memo[(r, c, snuke_index)] = False
        return False

    if dfs(0, 0, 0):
        print("Yes")
    else:
        print("No")

solve()