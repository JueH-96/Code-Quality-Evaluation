H, W = map(int, input().split())
S = [list(input()) for _ in range(H)]

for i in range(H):
    for j in range(W):
        if S[i][j] == "#":
            if i == 0 or S[i-1][j] == ".":
                if i == H-1 or S[i+1][j] == ".":
                    if j == 0 or S[i][j-1] == ".":
                        if j == W-1 or S[i][j+1] == ".":
                            print(i+1, j+1)
                            exit()
            if j == 0 or S[i][j-1] == ".":
                if j == W-1 or S[i][j+1] == ".":
                    if i == 0 or S[i-1][j] == ".":
                        if i == H-1 or S[i+1][j] == ".":
                            print(i+1, j+1)
                            exit()