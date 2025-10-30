N, M = map(int, input().split())
S = [input().strip() for _ in range(N)]
T = [input().strip() for _ in range(M)]

for a in range(1, N - M + 2):
    for b in range(1, N - M + 2):
        match = True
        for i in range(M):
            for j in range(M):
                s_row = a - 1 + i
                s_col = b - 1 + j
                if S[s_row][s_col] != T[i][j]:
                    match = False
                    break
            if not match:
                break
        if match:
            print(a, b)
            exit()