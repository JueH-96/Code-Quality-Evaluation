import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    is_empty = [[(c == '.') for c in row] for row in grid]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    total = 0

    for i in range(H):
        for j in range(W):
            if is_empty[i][j]:
                start_mask = 1 << (i * W + j)
                stack = [(i, j, 0, start_mask)]
                cnt = 0
                while stack:
                    current_i, current_j, steps, mask = stack.pop()
                    if steps == K:
                        cnt += 1
                        continue
                    for di, dj in dirs:
                        ni = current_i + di
                        nj = current_j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            if is_empty[ni][nj] and not (mask & (1 << (ni * W + nj))):
                                new_mask = mask | (1 << (ni * W + nj))
                                stack.append((ni, nj, steps + 1, new_mask))
                total += cnt
    print(total)

if __name__ == "__main__":
    main()