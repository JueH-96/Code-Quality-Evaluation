import sys

def read_input():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H_A, W_A = int(data[idx]), int(data[idx+1])
    idx += 2
    A = [data[i] for i in range(idx, idx+H_A)]
    idx += H_A
    H_B, W_B = int(data[idx]), int(data[idx+1])
    idx += 2
    B = [data[i] for i in range(idx, idx+H_B)]
    idx += H_B
    H_X, W_X = int(data[idx]), int(data[idx+1])
    idx += 2
    X = [data[i] for i in range(idx, idx+H_X)]
    return A, B, X, H_A, W_A, H_B, W_B, H_X, W_X

def can_create_sheet(A, B, X, H_A, W_A, H_B, W_B, H_X, W_X):
    def merge_sheets(sheet1, sheet2, h1, w1, h2, w2):
        merged = [['.' for _ in range(w1 + w2)] for _ in range(max(h1, h2))]
        for i in range(h1):
            for j in range(w1):
                if sheet1[i][j] == '#':
                    merged[i][j] = '#'
        for i in range(h2):
            for j in range(w2):
                if sheet2[i][j] == '#':
                    merged[i][j + w1] = '#'
        return merged

    def check_match(merged, X, h1, w1, h2, w2, hX, wX):
        for i in range(h1 + h2 - hX + 1):
            for j in range(w1 + w2 - wX + 1):
                match = True
                for x in range(hX):
                    for y in range(wX):
                        if merged[i + x][j + y] != X[x][y]:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    return True
        return False

    for i in range(H_A + 1):
        for j in range(W_A + 1):
            merged = merge_sheets(A, B, H_A, W_A, H_B, W_B)
            if check_match(merged, X, H_A, W_A, H_B, W_B, H_X, W_X):
                return "Yes"
    return "No"

A, B, X, H_A, W_A, H_B, W_B, H_X, W_X = read_input()
print(can_create_sheet(A, B, X, H_A, W_A, H_B, W_B, H_X, W_X))