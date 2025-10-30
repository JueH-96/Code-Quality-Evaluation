# YOUR CODE HERE
def can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X):
    def paste_sheet(C, sheet, H_sheet, W_sheet, x, y):
        for i in range(H_sheet):
            for j in range(W_sheet):
                if sheet[i][j] == '#':
                    C[x + i][y + j] = '#'

    def check_cutout(C, x, y, H_X, W_X, X):
        for i in range(H_X):
            for j in range(W_X):
                if C[x + i][y + j] != X[i][j]:
                    return False
        return True

    # Create an infinitely large sheet C (big enough to cover all possible placements of A and B)
    max_size = 20  # Since H_A, W_A, H_B, W_B, H_X, W_X <= 10, 20 is more than enough
    C = [['.' for _ in range(max_size)] for _ in range(max_size)]

    # Try all possible placements of A and B on C
    for ax in range(max_size - H_A + 1):
        for ay in range(max_size - W_A + 1):
            for bx in range(max_size - H_B + 1):
                for by in range(max_size - W_B + 1):
                    # Reset C to all transparent squares
                    C = [['.' for _ in range(max_size)] for _ in range(max_size)]
                    
                    # Paste A and B onto C
                    paste_sheet(C, A, H_A, W_A, ax, ay)
                    paste_sheet(C, B, H_B, W_B, bx, by)
                    
                    # Try all possible cutouts of size H_X x W_X from C
                    for cx in range(max_size - H_X + 1):
                        for cy in range(max_size - W_X + 1):
                            if check_cutout(C, cx, cy, H_X, W_X, X):
                                return "Yes"
    return "No"

import sys
input = sys.stdin.read().split()
index = 0

H_A = int(input[index])
W_A = int(input[index + 1])
index += 2
A = input[index:index + H_A]
index += H_A

H_B = int(input[index])
W_B = int(input[index + 1])
index += 2
B = input[index:index + H_B]
index += H_B

H_X = int(input[index])
W_X = int(input[index + 1])
index += 2
X = input[index:index + H_X]

print(can_create_sheet_x(H_A, W_A, A, H_B, W_B, B, H_X, W_X, X))