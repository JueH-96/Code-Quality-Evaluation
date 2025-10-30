def solve():
    H_A, W_A = map(int, input().split())
    A = [list(input()) for _ in range(H_A)]
    H_B, W_B = map(int, input().split())
    B = [list(input()) for _ in range(H_B)]
    H_X, W_X = map(int, input().split())
    X = [list(input()) for _ in range(H_X)]

    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                for dx in range(H_B):
                    for dy in range(W_B):
                        if B[dx][dy] == '#':
                            if 0 <= i + dx < H_X and 0 <= j + dy < W_X:
                                if X[i + dx][j + dy] == '#':
                                    X[i + dx][j + dy] = '.'
                                else:
                                    print('No')
                                    return
                            else:
                                print('No')
                                return
    for i in range(H_X):
        for j in range(W_X):
            if X[i][j] == '#':
                print('No')
                return
    print('Yes')

solve()