N = int(input())
A = [list(input()) for _ in range(N)]

def shift_grid(A):
    N = len(A)
    B = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            B[i][j] = A[(i+1)%N][(j+1)%N]
    return B

B = shift_grid(A)
for row in B:
    print(''.join(str(i) for i in row))