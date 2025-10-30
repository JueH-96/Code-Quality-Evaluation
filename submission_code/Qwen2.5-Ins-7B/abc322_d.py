# YOUR CODE HERE
from itertools import permutations, product

def rotate(p):
    return tuple(zip(*p[::-1]))

def flip(p):
    return tuple(reversed(p[0])) + tuple(reversed(p[1:])) + tuple(reversed(p[4:6])) + tuple(reversed(p[7:9]))

def check(p1, p2, p3):
    grid = [('#' * 4,) * 4] * 4
    for p in p1 + p2 + p3:
        for r in range(4):
            for f in [0, 1]:
                if all(grid[i][j] == '.' for i, j in product(range(4), range(4)) if (i, j) in p):
                    for i, j in p:
                        grid[i][j] = '#'
                    return True
                p = flip(p) if f else rotate(p)
    return False

def solve():
    p1 = tuple(input().strip() for _ in range(4))
    p2 = tuple(input().strip() for _ in range(4))
    p3 = tuple(input().strip() for _ in range(4))
    return 'Yes' if check(p1, p2, p3) else 'No'

print(solve())