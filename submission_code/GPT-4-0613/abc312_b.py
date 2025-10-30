N, M = map(int, input().split())
S = [list(input()) for _ in range(N)]

dx = [0, 0, 0, -1, -1, -1, 1, 1, 1]
dy = [0, -1, 1, 0, -1, 1, 0, -1, 1]

for i in range(N):
    for j in range(M):
        if S[i][j] == '#':
            for k in range(9):
                nx, ny = i + dx[k], j + dy[k]
                if nx < 0 or ny < 0 or nx >= N or ny >= M or S[nx][ny] == '#':
                    break
            else:
                if i - 3 >= 0 and j - 3 >= 0 and i + 3 < N and j + 3 < M:
                    for x in range(i - 3, i + 4):
                        for y in range(j - 3, j + 4):
                            if x < i - 1 or x > i + 1 or y < j - 1 or y > j + 1:
                                if S[x][y] == '#':
                                    break
                        else:
                            continue
                        break
                    else:
                        print(i - 2, j - 2)