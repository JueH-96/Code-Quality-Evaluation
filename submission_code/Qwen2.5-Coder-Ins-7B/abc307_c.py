# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H_A, W_A = map(int, data[0:2])
A = data[2:2+H_A]
H_B, W_B = map(int, data[2+H_A:4+H_A])
B = data[4+H_A:4+H_A+H_B]
H_X, W_X = map(int, data[4+H_A+H_B:6+H_A+H_B])
X = data[6+H_A+H_B:]

def is_valid_cut(A, B, X, H_A, W_A, H_B, W_B, H_X, W_X):
    for i in range(H_A + H_B - H_X + 1):
        for j in range(W_A + W_B - W_X + 1):
            cut = [['.' for _ in range(W_X)] for _ in range(H_X)]
            for x in range(H_A):
                for y in range(W_A):
                    if A[x][y] == '#':
                        cut[x][y] = '#'
            for x in range(H_B):
                for y in range(W_B):
                    if B[x][y] == '#':
                        cut[x + i][y + j] = '#'
            if cut == X:
                return True
    return False

if is_valid_cut(A, B, X, H_A, W_A, H_B, W_B, H_X, W_X):
    print("Yes")
else:
    print("No")