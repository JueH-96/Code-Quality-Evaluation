# YOUR CODE HERE
def can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X):
    def is_valid_cutout(A, B, X, ax, ay, bx, by):
        for i in range(H_X):
            for j in range(W_X):
                x_char = X[i][j]
                a_char = A[ax + i][ay + j] if 0 <= ax + i < H_A and 0 <= ay + j < W_A else '.'
                b_char = B[bx + i][by + j] if 0 <= bx + i < H_B and 0 <= by + j < W_B else '.'
                if x_char == '#':
                    if a_char != '#' and b_char != '#':
                        return False
        return True

    for ax in range(-H_A + 1, H_X):
        for ay in range(-W_A + 1, W_X):
            for bx in range(-H_B + 1, H_X):
                for by in range(-W_B + 1, W_X):
                    if is_valid_cutout(A, B, X, ax, ay, bx, by):
                        return "Yes"
    return "No"

import sys
input = sys.stdin.read
data = input().split()

H_A = int(data[0])
W_A = int(data[1])
A = data[2:2 + H_A]

H_B = int(data[2 + H_A])
W_B = int(data[3 + H_A])
B = data[4 + H_A:4 + H_A + H_B]

H_X = int(data[4 + H_A + H_B])
W_X = int(data[5 + H_A + H_B])
X = data[6 + H_A + H_B:6 + H_A + H_B + H_X]

print(can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X))