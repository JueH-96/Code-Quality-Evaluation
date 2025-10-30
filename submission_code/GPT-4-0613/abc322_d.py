def solve():
    P = [list(input()) for _ in range(12)]
    C = [0]*16
    for i in range(12):
        for j in range(4):
            if P[i][j] == '#':
                C[i//4*4+j//4] += 1
    C.sort()
    if C == [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5]:
        print('Yes')
    else:
        print('No')

solve()