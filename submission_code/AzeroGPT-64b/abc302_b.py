def check_snuke(r, c, H, W, S):
    dr = [-1, 0, 1, 0, 0]
    dc = [0, 1, 0, -1, 0]
    for i in range(4):
        ok = True
        for j in range(5):
            nr, nc = r + dr[j] * j, c + dc[j] * j
            if not (0 <= nr < H and 0 <= nc < W) or S[nr][nc] != "snuke"[j]:
                ok = False
                break
        if ok:
            for j in range(5):
                print(nr, nc)
            return True
    dr, dc = [-1, -1, -1, 1, 1, 1], [0, 1, -1, -1, 1, 0]
    for i in range(6):
        ok = True
        for j in range(5):
            nr, nc = r + dr[i] * j, c + dc[i] * j
            if not (0 <= nr < H and 0 <= nc < W) or S[nr][nc] != "snuke"[j]:
                ok = False
                break
        if ok:
            for j in range(5):
                print(nr, nc)
            return True
    return False

def main():
    H, W = map(int, input().split())
    S = [input() for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if S[r][c] == "s":
                if check_snuke(r, c, H, W, S):
                    return

main()