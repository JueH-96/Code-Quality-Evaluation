H, W, N = map(int, input().split())
T = input()
S = [input() for _ in range(H)]

def count_cells(H, W, T, S):
    cells = set()
    for i in range(H):
        for j in range(W):
            if T[i] == 'L' and j > 0 and S[i][j-1] == '.':
                cells.add((i, j-1))
            if T[i] == 'R' and j < W-1 and S[i][j] == '.':
                cells.add((i, j))
            if T[i] == 'U' and i > 0 and S[i-1][j] == '.':
                cells.add((i-1, j))
            if T[i] == 'D' and i < H-1 and S[i][j] == '.':
                cells.add((i, j))
    return len(cells)

print(count_cells(H, W, T, S))