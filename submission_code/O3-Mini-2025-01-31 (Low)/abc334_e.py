def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(10**7)
    mod = 998244353

    data = sys.stdin.read().split()
    if not data:
        return

    H, W = int(data[0]), int(data[1])
    grid = data[2:]
    
    # comp[i][j] will store the component id of green cell at (i,j) (if any)
    comp = [[-1] * W for _ in range(H)]
    
    comp_id = 0
    # Find connected components among the initially green cells.
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp[i][j] == -1:
                dq = deque()
                dq.append((i, j))
                comp[i][j] = comp_id
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp[nx][ny] == -1:
                                comp[nx][ny] = comp_id
                                dq.append((nx, ny))
                comp_id += 1

    # Let c be the total number of green connected components currently.
    c = comp_id
    
    # We now iterate over each red cell.
    # When repainting a red cell to green, 
    # if none of its 4-neighbors is green, it creates a new component (+1).
    # If it is adjacent to k distinct green components, the new cell merges them,
    # and the net effect is c - (k - 1). (When k = 1, then the number remains c.)
    total_value = 0
    red_count = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_count += 1
                neighbor_components = set()
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                        neighbor_components.add(comp[nx][ny])
                if not neighbor_components:
                    new_components = c + 1
                else:
                    new_components = c - (len(neighbor_components) - 1)
                total_value += new_components

    # The expected value E is (total_value) / (red_count).
    # As required, we need to output the unique integer R such that R * red_count ≡ total_value (mod mod).
    # That is, we compute: R = total_value * inv(red_count) mod mod.
    inv_red = pow(red_count, mod - 2, mod)
    ans = (total_value % mod) * inv_red % mod

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()