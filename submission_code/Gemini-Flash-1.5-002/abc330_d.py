# YOUR CODE HERE
n = int(input())
grid = [input() for _ in range(n)]

count = 0
for r1 in range(n):
    for c1 in range(n):
        for r2 in range(n):
            for c2 in range(n):
                for r3 in range(n):
                    for c3 in range(n):
                        if (r1, c1) == (r2, c2) or \
                           (r1, c1) == (r3, c3) or \
                           (r2, c2) == (r3, c3):
                            continue
                        if grid[r1][c1] == 'o' and \
                           grid[r2][c2] == 'o' and \
                           grid[r3][c3] == 'o':
                            if (r1 == r2 or r1 == r3 or r2 == r3) and \
                               (c1 == c2 or c1 == c3 or c2 == c3):
                                if (r1 == r2 and r1 == r3) or \
                                   (c1 == c2 and c1 == c3):
                                    continue
                                if (r1 == r2 and c1 == c3) or \
                                   (r1 == r2 and c2 == c3) or \
                                   (r1 == r3 and c1 == c2) or \
                                   (r1 == r3 and c2 == c3) or \
                                   (r2 == r3 and c1 == c2) or \
                                   (r2 == r3 and c1 == c3):
                                    count += 1

print(count // 6)