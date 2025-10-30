H_A, W_A = map(int, input().split())
A = [input().strip() for _ in range(H_A)]
H_B, W_B = map(int, input().split())
B = [input().strip() for _ in range(H_B)]
H_X, W_X = map(int, input().split())
X = [input().strip() for _ in range(H_X)]

a_blacks = []
for i in range(H_A):
    for j in range(W_A):
        if A[i][j] == '#':
            a_blacks.append((i, j))
b_blacks = []
for i in range(H_B):
    for j in range(W_B):
        if B[i][j] == '#':
            b_blacks.append((i, j))

if not a_blacks or not b_blacks:
    print("No")
    exit()

min_a_row = min(i for i, j in a_blacks)
max_a_row = max(i for i, j in a_blacks)
min_a_col = min(j for i, j in a_blacks)
max_a_col = max(j for i, j in a_blacks)
min_b_row = min(i for i, j in b_blacks)
max_b_row = max(i for i, j in b_blacks)
min_b_col = min(j for i, j in b_blacks)
max_b_col = max(j for i, j in b_blacks)

x0_min = (min_a_row + min_b_row) - H_X
x0_max = (max_a_row + max_b_row) + H_X
y0_min = (min_a_col + min_b_col) - W_X
y0_max = (max_a_col + max_b_col) + W_X

for x0 in range(x0_min, x0_max + 1):
    for y0 in range(y0_min, y0_max + 1):
        dx_min = max(x0 - i for i, j in a_blacks)
        dx_max = min(x0 + H_X - 1 - i for i, j in a_blacks)
        if dx_min > dx_max:
            continue
        dy_min = max(y0 - j for i, j in a_blacks)
        dy_max = min(y0 + W_X - 1 - j for i, j in a_blacks)
        if dy_min > dy_max:
            continue
        ew_min = max(x0 - k for k, l in b_blacks)
        ew_max = min(x0 + H_X - 1 - k for k, l in b_blacks)
        if ew_min > ew_max:
            continue
        nx_min = max(y0 - l for k, l in b_blacks)
        nx_max = min(y0 + W_X - 1 - l for k, l in b_blacks)
        if nx_min > nx_max:
            continue
        
        for dx in range(dx_min, dx_max + 1):
            for dy in range(dy_min, dy_max + 1):
                for ew in range(ew_min, ew_max + 1):
                    for nx in range(nx_min, nx_max + 1):
                        gen_sheet = [['.' for _ in range(W_X)] for _ in range(H_X)]
                        for i, j in a_blacks:
                            x = i + dx
                            y = j + dy
                            gen_sheet[x - x0][y - y0] = '#'
                        for k, l in b_blacks:
                            x = k + ew
                            y = l + nx
                            gen_sheet[x - x0][y - y0] = '#'
                        match = True
                        for i in range(H_X):
                            if gen_sheet[i] != X[i]:
                                match = False
                                break
                        if match:
                            print("Yes")
                            exit()

print("No")