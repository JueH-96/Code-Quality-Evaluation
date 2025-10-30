# YOUR CODE HERE
N = int(input())
grid_A = [input().strip() for _ in range(N)]
grid_B = [input().strip() for _ in range(N)]

for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i+1, j+1)
            exit()