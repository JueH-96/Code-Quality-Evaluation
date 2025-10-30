H, W, N = map(int, input().split())
T = input()
S = [list(input()) for _ in range(H)]

dx = {'U': -1, 'D': 1, 'L': 0, 'R': 0}
dy = {'U': 0, 'D': 0, 'L': -1, 'R': 1}

X = [0]*4
Y = [0]*4
X[0] = X[2] = 0
X[1] = X[3] = H-1
Y[0] = Y[3] = 0
Y[1] = Y[2] = W-1

for i in range(N-1, -1, -1):
    if T[i] == 'U':
        while X[0] < H and S[X[0]][Y[0]] == '#':
            X[0] += 1
        X[1] = min(X[1]+1, X[0])
    if T[i] == 'D':
        while X[3] >= 0 and S[X[3]][Y[3]] == '#':
            X[3] -= 1
        X[2] = max(X[2]-1, X[3])
    if T[i] == 'L':
        while Y[0] < W and S[X[0]][Y[0]] == '#':
            Y[0] += 1
        Y[1] = min(Y[1]+1, Y[0])
    if T[i] == 'R':
        while Y[3] >= 0 and S[X[3]][Y[3]] == '#':
            Y[3] -= 1
        Y[2] = max(Y[2]-1, Y[3])

    if X[0] > X[1] or Y[0] > Y[1]:
        print(0)
        exit(0)

    X[0] = max(0, X[0]-dx[T[i]])
    X[1] = min(H-1, X[1]-dx[T[i]])
    Y[0] = max(0, Y[0]-dy[T[i]])
    Y[1] = min(W-1, Y[1]-dy[T[i]])

    if X[0] > X[1] or Y[0] > Y[1]:
        print(0)
        exit(0)

print((X[1]-X[0]+1)*(Y[1]-Y[0]+1))