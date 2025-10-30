# YOUR CODE HERE
H_A, W_A = map(int, input().split())
A = [input() for _ in range(H_A)]
H_B, W_B = map(int, input().split())
B = [input() for _ in range(H_B)]
H_X, W_X = map(int, input().split())
X = [input() for _ in range(H_X)]

def check(A, B, X):
    for i in range(H_X):
        for j in range(W_X):
            if X[i][j] == '#':
                if i >= H_A or j >= W_A or A[i][j] == '.':
                    if i >= H_B or j >= W_B or B[i][j] == '.':
                        return False
    return True

print('Yes' if check(A, B, X) else 'No')