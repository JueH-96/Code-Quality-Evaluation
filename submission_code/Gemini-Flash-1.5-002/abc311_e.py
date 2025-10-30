# YOUR CODE HERE
H, W, N = map(int, input().split())
holes = []
for i in range(N):
    a, b = map(int, input().split())
    holes.append((a, b))

grid = [[0 for _ in range(W)] for _ in range(H)]
for hole in holes:
    grid[hole[0] - 1][hole[1] - 1] = 1

count = 0
for i in range(H):
    for j in range(W):
        for n in range(1, min(H - i, W - j) + 1):
            holeless = True
            for k in range(n):
                for l in range(n):
                    if grid[i + k][j + l] == 1:
                        holeless = False
                        break
                if not holeless:
                    break
            if holeless:
                count += 1

print(count)