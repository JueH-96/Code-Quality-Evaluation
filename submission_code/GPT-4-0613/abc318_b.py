N = int(input().strip())
sheets = [list(map(int, input().split())) for _ in range(N)]

plane = [[0]*101 for _ in range(101)]

for sheet in sheets:
    A, B, C, D = sheet
    for i in range(A, B):
        for j in range(C, D):
            plane[i][j] = 1

S = sum(sum(row) for row in plane)
print(S)