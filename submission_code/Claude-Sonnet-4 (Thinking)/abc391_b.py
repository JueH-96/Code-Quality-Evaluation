N, M = map(int, input().split())

S = []
for i in range(N):
    S.append(input().strip())

T = []
for i in range(M):
    T.append(input().strip())

# Find the matching position
for start_r in range(N - M + 1):
    for start_c in range(N - M + 1):
        match = True
        for i in range(M):
            for j in range(M):
                if S[start_r + i][start_c + j] != T[i][j]:
                    match = False
                    break
            if not match:
                break
        if match:
            print(start_r + 1, start_c + 1)
            exit()