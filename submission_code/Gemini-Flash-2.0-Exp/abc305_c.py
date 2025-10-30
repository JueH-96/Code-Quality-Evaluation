H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())

for r1 in range(H):
    for c1 in range(W):
        for r2 in range(r1 + 1, H):
            for c2 in range(c1 + 1, W):
                count = 0
                empty_r = -1
                empty_c = -1
                valid = True
                for i in range(H):
                    for j in range(W):
                        if r1 <= i <= r2 and c1 <= j <= c2:
                            if grid[i][j] == '#':
                                count += 1
                            else:
                                if empty_r == -1:
                                    empty_r = i
                                    empty_c = j
                                count += 0
                        else:
                            if grid[i][j] == '#':
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    expected_count = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if expected_count - 1 == count and empty_r != -1:
                        print(empty_r + 1, empty_c + 1)
                        exit()