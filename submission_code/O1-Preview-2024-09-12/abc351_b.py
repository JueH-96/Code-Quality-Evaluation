# YOUR CODE HERE
n = int(input())
A = [input() for _ in range(n)]
B = [input() for _ in range(n)]
for i in range(n):
    for j in range(n):
        if A[i][j] != B[i][j]:
            print(i+1, j+1)
            exit()