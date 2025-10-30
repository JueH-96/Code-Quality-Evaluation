from itertools import product

def rotate(P):
    return [''.join([P[j][i] for j in range(3, -1, -1)]) for i in range(4)]

def check(P, Q, R):
    grid = [['.' for _ in range(4)] for _ in range(4)]
    for i, j in product(range(4), range(4)):
        if P[i][j] == '#' and grid[i][j] != '.':
            return False
        grid[i][j] = P[i][j]
    for i, j in product(range(4), range(4)):
        if Q[i][j] == '#' and grid[i][j] != '.':
            return False
        grid[i][j] = Q[i][j]
    for i, j in product(range(4), range(4)):
        if R[i][j] == '#' and grid[i][j] != '.':
            return False
        grid[i][j] = R[i][j]
    return all(grid[i][j] == '#' for i, j in product(range(4), range(4)))

P = [input() for _ in range(4)]
Q = [input() for _ in range(4)]
R = [input() for _ in range(4)]

for _ in range(4):
    for _ in range(4):
        for _ in range(4):
            if check(P, Q, R):
                print('Yes')
                exit()
            Q = rotate(Q)
        R = rotate(R)
    P = rotate(P)
print('No')