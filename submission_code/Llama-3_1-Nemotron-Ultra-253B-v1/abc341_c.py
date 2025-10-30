H, W, N = map(int, input().split())
T = input().strip()
S = [input().strip() for _ in range(H)]

delta = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
result = set()

for i in range(1, H + 1):
    for j in range(1, W + 1):
        if S[i-1][j-1] != '.':
            continue
        ci, cj = i, j
        valid = True
        for move in T:
            di, dj = delta[move]
            ci += di
            cj += dj
            if 1 <= ci <= H and 1 <= cj <= W:
                if S[ci-1][cj-1] != '.':
                    valid = False
                    break
            else:
                valid = False
                break
        if valid:
            result.add((ci, cj))

print(len(result))