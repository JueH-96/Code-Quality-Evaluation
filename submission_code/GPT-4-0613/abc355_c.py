# YOUR CODE HERE
N, T = map(int, input().split())
A = list(map(int, input().split()))

row = [0]*N
col = [0]*N
diag = [0, 0]
for i in range(T):
    a = A[i]-1
    x, y = divmod(a, N)
    row[x] += 1
    col[y] += 1
    if x == y:
        diag[0] += 1
    if x == N-1-y:
        diag[1] += 1
    if row[x] == N or col[y] == N or diag[0] == N or diag[1] == N:
        print(i+1)
        break
else:
    print(-1)