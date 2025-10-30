H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

hashes = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            hashes.append((i, j))

K = len(hashes)
if K == 0:
    print(1, 1)
    exit()

min_row_h = min(i for i, j in hashes)
max_row_h = max(i for i, j in hashes)
min_col_h = min(j for i, j in hashes)
max_col_h = max(j for i, j in hashes)

# Precompute prefix sums for '#' and '.'
hash_prefix = [[0] * (W + 1) for _ in range(H + 1)]
dot_prefix = [[0] * (W + 1) for _ in range(H + 1)]

for i in range(H):
    for j in range(W):
        hash_prefix[i+1][j+1] = hash_prefix[i+1][j] + hash_prefix[i][j+1] - hash_prefix[i][j]
        hash_prefix[i+1][j+1] += 1 if grid[i][j] == '#' else 0
        dot_prefix[i+1][j+1] = dot_prefix[i+1][j] + dot_prefix[i][j+1] - dot_prefix[i][j]
        dot_prefix[i+1][j+1] += 1 if grid[i][j] == '.' else 0

for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            min_row = min(min_row_h, i)
            max_row = max(max_row_h, i)
            min_col = min(min_col_h, j)
            max_col = max(max_col_h, j)
            
            area = (max_row - min_row + 1) * (max_col - min_col + 1)
            if area != K + 1:
                continue
            
            a = min_row + 1
            b = max_row + 1
            c = min_col + 1
            d = max_col + 1
            
            hash_count = hash_prefix[b][d] - hash_prefix[a-1][d] - hash_prefix[b][c-1] + hash_prefix[a-1][c-1]
            if hash_count != (area - 1):
                continue
            
            dot_count = dot_prefix[b][d] - dot_prefix[a-1][d] - dot_prefix[b][c-1] + dot_prefix[a-1][c-1]
            if dot_count != 1:
                continue
            
            print(i + 1, j + 1)
            exit()