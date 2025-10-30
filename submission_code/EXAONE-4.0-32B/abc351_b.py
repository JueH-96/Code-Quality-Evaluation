n = int(input().strip())
grid_A = [input().strip() for _ in range(n)]
grid_B = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if grid_A[i][j] != grid_B[i][j]:
            print(f"{i+1} {j+1}")
            exit(0)