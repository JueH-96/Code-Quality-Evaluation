H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
S_i -= 1
S_j -= 1
C = [list(input()) for _ in range(H)]
X = input()

for i in X:
    if i == 'L' and S_j - 1 >= 0 and C[S_i][S_j - 1] == '.':
        S_j -= 1
    elif i == 'R' and S_j + 1 < W and C[S_i][S_j + 1] == '.':
        S_j += 1
    elif i == 'U' and S_i - 1 >= 0 and C[S_i - 1][S_j] == '.':
        S_i -= 1
    elif i == 'D' and S_i + 1 < H and C[S_i + 1][S_j] == '.':
        S_i += 1

print(S_i + 1, S_j + 1)