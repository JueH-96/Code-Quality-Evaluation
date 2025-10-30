from collections import defaultdict
import sys
input = sys.stdin.readline

H, W = map(int, input().split())
A = [input().strip() for _ in range(H)]

# check the first character
if A[0][0] != 's' or A[H-1][W-1] != 'e':
    print('No')
    sys.exit()

# define the path pattern
path_pattern = 'snuke' * (H + W)

visited = [[False] * W for _ in range(H)]
queue = [(0, 0, A[0][0], 0)]  # (i, j, current character, index in path pattern)

while queue:
    i, j, prev_char, path_idx = queue.pop(0)

    if i == H-1 and j == W-1 and prev_char == 'e':
        print('Yes')
        sys.exit()

    for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
            if A[ni][nj] == path_pattern[path_idx]:
                visited[ni][nj] = True
                queue.append((ni, nj, A[ni][nj], (path_idx + 1) % len(path_pattern)))

print('No')