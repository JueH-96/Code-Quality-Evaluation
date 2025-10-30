import sys
import threading

def main():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().rstrip() for _ in range(H)]
    # Directions: 8 neighbors
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    target = "snuke"
    for i in range(H):
        for j in range(W):
            if S[i][j] != 's':
                continue
            for dx, dy in directions:
                coords = [(i, j)]
                x, y = i, j
                ok = True
                for k in range(1, 5):
                    x += dx
                    y += dy
                    if not (0 <= x < H and 0 <= y < W and S[x][y] == target[k]):
                        ok = False
                        break
                    coords.append((x, y))
                if ok:
                    for (r, c) in coords:
                        print(r+1, c+1)
                    return

if __name__ == "__main__":
    main()