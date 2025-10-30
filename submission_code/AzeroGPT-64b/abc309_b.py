N = int(input())
A = [list(map(int, list(input()))) for _ in range(N)]
B = [[0 for _ in range(N)] for _ in range(N)]

# Outermost layer
for i in range(N):
    B[i][N-1] = A[i][N-2]
for i in range(1, N):
    B[N-1][N-i-1] = A[N-i-1][N-1]
for i in range(1, N):
    B[N-i-1][0] = A[0][i]
B[0][1:N-1] = A[1][1:N-1]

if N >= 3:
    # Inner layers
    rows = A[1][1:N-1]
    cols = []
    for i in range(N-2):
        cols.append(A[i+1][1])
        cols.append(A[i+1][N-2])
        A[i+1][1] = cols[-2]
        A[i+1][N-2] = cols[-2]
    cols.append(A[N-2][1])
    rows.reverse()
    cols.reverse()
    for i in range(1, N-1):
        A[i][N-2] = rows[i-1]
    for i in range(1, N-1):
        A[i][1] = cols[N-3-i]
    for i in range(N-2):
        B[i+1][1] = cols[i]
        B[i+1][N-2] = cols[-(i+1)]
    B[1][1:N-2] = rows

for row in B:
    print("".join(map(str, row)))