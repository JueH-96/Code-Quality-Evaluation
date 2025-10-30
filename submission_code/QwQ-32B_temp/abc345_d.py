import sys

def main():
    N, H, W = map(int, sys.stdin.readline().split())
    tiles = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        tiles.append((a, b))
    target_area = H * W
    found = False

    for mask in range(1, 1 << N):
        subset = []
        total_area = 0
        for i in range(N):
            if (mask >> i) & 1:
                a, b = tiles[i]
                subset.append((a, b))
                total_area += a * b
        if total_area != target_area:
            continue

        subset.sort(key=lambda x: -(x[0] * x[1]))
        grid = [[False for _ in range(W)] for _ in range(H)]
        used = [False] * len(subset)
        found_flag = False

        def backtrack():
            nonlocal found_flag
            r, c = -1, -1
            for i in range(H):
                for j in range(W):
                    if not grid[i][j]:
                        r, c = i, j
                        break
                if r != -1:
                    break
            if r == -1:
                found_flag = True
                return

            for i in range(len(subset)):
                if used[i]:
                    continue
                a, b = subset[i]
                for h, w in [(a, b), (b, a)]:
                    if r + h > H or c + w > W:
                        continue
                    valid = True
                    for dr in range(h):
                        for dc in range(w):
                            if grid[r + dr][c + dc]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        for dr in range(h):
                            for dc in range(w):
                                grid[r + dr][c + dc] = True
                        used[i] = True
                        backtrack()
                        if found_flag:
                            return
                        for dr in range(h):
                            for dc in range(w):
                                grid[r + dr][c + dc] = False
                        used[i] = False

        backtrack()
        if found_flag:
            found = True
            break

    print("Yes" if found else "No")

if __name__ == "__main__":
    main()