import sys

def read_input():
    H_A = int(sys.stdin.readline().split()[0])
    W_A = int(sys.stdin.readline().split()[0])
    A = [sys.stdin.readline().strip() for _ in range(H_A)]
    H_B = int(sys.stdin.readline().split()[0])
    W_B = int(sys.stdin.readline().split()[0])
    B = [sys.stdin.readline().strip() for _ in range(H_B)]
    H_X = int(sys.stdin.readline().split()[0])
    W_X = int(sys.stdin.readline().split()[0])
    X = [sys.stdin.readline().strip() for _ in range(H_X)]
    return H_A, W_A, A, H_B, W_B, B, H_X, W_X, X

def solve():
    H_A, W_A, A, H_B, W_B, B, H_X, W_X, X = read_input()
    C = [['.' for _ in range(W_A + W_B)] for _ in range(H_A + H_B)]
    for i in range(H_A):
        for j in range(W_A):
            if A[i][j] == '#':
                C[i][j] = '#'
    for i in range(H_B):
        for j in range(W_B):
            if B[i][j] == '#':
                C[i+H_A][j] = '#'
    for i in range(H_X):
        for j in range(W_X):
            if X[i][j] == '#' and C[i][j] == '.':
                print('No')
                return
    print('Yes')

solve()