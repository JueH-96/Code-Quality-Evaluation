def solve():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    target = "snuke"

    for r in range(H):
        for c in range(W):
            if S[r][c] == 's':
                for dr, dc in directions:
                    path = []
                    found = True
                    for i in range(5):
                        nr = r + i * dr
                        nc = c + i * dc
                        if 0 <= nr < H and 0 <= nc < W and S[nr][nc] == target[i]:
                            path.append((nr + 1, nc + 1))
                        else:
                            found = False
                            break
                    if found:
                        for row, col in path:
                            print(row, col)
                        return

solve()