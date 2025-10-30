# YOUR CODE HERE
def find_difference():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    grid_A = data[1:N+1]
    grid_B = data[N+1:2*N+1]
    
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                print(i + 1, j + 1)
                return

find_difference()