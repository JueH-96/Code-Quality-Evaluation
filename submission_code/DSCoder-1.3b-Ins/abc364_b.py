H, W = map(int, input().split())
S = list(map(int, input().split()))
C = [list(input()) for _ in range(H)]
X = input()

def move(i, j, direction):
    if direction == 'U' and i > 0 and C[i-1][j] == '.':
        return i-1, j
    elif direction == 'D' and i < H-1 and C[i+1][j] == '.':
        return i+1, j
    elif direction == 'L' and j > 0 and C[i][j-1] == '.':
        return i, j-1
    elif direction == 'R' and j < W-1 and C[i][j+1] == '.':
        return i, j+1
    else:
        return i, j

i, j = S[0]-1, S[1]-1
for direction in X:
    i, j = move(i, j, direction)

print(i+1, j+1)