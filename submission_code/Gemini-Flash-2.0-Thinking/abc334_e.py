def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]

    red_cells = []
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                red_cells.append((i, j))

    num_red_cells = len(red_cells)
    if num_red_cells == 0:
        print(0)
        return

    MOD = 998244353

    def count_components(grid_state):
        components = 0
        visited = [[False for _ in range(W)] for _ in range(H)]
        for i in range(H):
            for j in range(W):
                if grid_state[i][j] == '#' and not visited[i][j]:
                    components += 1
                    stack = [(i, j)]
                    visited[i][j] = True
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and grid_state[nr][nc] == '#' and not visited[nr][nc]:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
        return components

    sum_components = 0
    for r_r, r_c in red_cells:
        temp_grid_list = [list(row) for row in S]
        temp_grid_list[r_r][r_c] = '#'
        temp_grid = ["".join(row) for row in temp_grid_list]
        sum_components += count_components(temp_grid)

    def power(a, b):
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def inverse(n):
        return power(n, MOD - 2)

    expected_numerator = sum_components % MOD
    expected_denominator_inv = inverse(num_red_cells)
    expected_value = (expected_numerator * expected_denominator_inv) % MOD
    print(expected_value)

solve()