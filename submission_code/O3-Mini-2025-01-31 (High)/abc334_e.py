def main():
    import sys
    from collections import deque
    input = sys.stdin.readline
    mod = 998244353

    # Read the input.
    # In the grid, '.' denotes a red cell and '#' denotes a green cell.
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # We want to compute the initial number of connected components among green cells.
    # We'll label each green connected component with an id.
    comp = [[-1] * W for _ in range(H)]
    comp_cnt = 0  # number of green connected components
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp[i][j] == -1:
                dq = deque()
                dq.append((i, j))
                comp[i][j] = comp_cnt
                while dq:
                    x, y = dq.popleft()
                    # Explore all 4 adjacent cells.
                    if x > 0 and grid[x - 1][y] == '#' and comp[x - 1][y] == -1:
                        comp[x - 1][y] = comp_cnt
                        dq.append((x - 1, y))
                    if x < H - 1 and grid[x + 1][y] == '#' and comp[x + 1][y] == -1:
                        comp[x + 1][y] = comp_cnt
                        dq.append((x + 1, y))
                    if y > 0 and grid[x][y - 1] == '#' and comp[x][y - 1] == -1:
                        comp[x][y - 1] = comp_cnt
                        dq.append((x, y - 1))
                    if y < W - 1 and grid[x][y + 1] == '#' and comp[x][y + 1] == -1:
                        comp[x][y + 1] = comp_cnt
                        dq.append((x, y + 1))
                comp_cnt += 1

    # Let c be the number of connected green components.
    c = comp_cnt

    # Now, we imagine repainting one red cell (cell with '.') to green uniformly at random.
    # For a red cell r, let its distinct adjacent (via up/down/left/right) green components count be d.
    # After repainting, if r touches no green cell (d == 0), it becomes a new green component so:
    #   f(r) = c + 1.
    # Otherwise, if there are d >= 1 distinct adjacent green components, then they merge (with r joining)
    # into one component; hence the total number is f(r) = c - (d - 1) = c + 1 - d.
    # Thus, for every red cell, f(r) = c + 1 - d.
    # We need to compute the expected value:
    #
    #   E = [Σ_{red cell r} (c + 1 - d(r))] / (# red cells).
    
    red_count = 0
    sum_d = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_count += 1
                seen = set()
                if i > 0 and grid[i - 1][j] == '#':
                    seen.add(comp[i - 1][j])
                if i < H - 1 and grid[i + 1][j] == '#':
                    seen.add(comp[i + 1][j])
                if j > 0 and grid[i][j - 1] == '#':
                    seen.add(comp[i][j - 1])
                if j < W - 1 and grid[i][j + 1] == '#':
                    seen.add(comp[i][j + 1])
                sum_d += len(seen)
    
    # The numerator of the expected value is:
    #    (#red cells) * (c + 1) - sum_d.
    # So the expected value (which is rational) is:
    #    E = (red_count * (c + 1) - sum_d) / red_count.
    #
    # The problem asks us to output the value modulo 998244353.
    # Since the expected value equals a fraction P/Q in lowest terms and Q is invertible modulo 998244353,
    # we compute R = (P * Q^{-1}) mod 998244353.
    
    numerator = red_count * (c + 1) - sum_d
    numerator %= mod
    inv_red = pow(red_count, mod - 2, mod)
    ans = (numerator * inv_red) % mod
    sys.stdout.write(str(ans) + "
")
    
if __name__ == '__main__':
    main()