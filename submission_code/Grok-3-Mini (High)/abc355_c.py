import sys
data = sys.stdin.read().split()
N = int(data[0])
T = int(data[1])
A_list = list(map(int, data[2:2+T]))

row_count = [0] * N
col_count = [0] * N
main_diag_count = 0
anti_diag_count = 0

bingo_turn = -1
for turn, A in enumerate(A_list, start=1):
    row_idx = (A - 1) // N
    col_idx = (A - 1) % N
    
    # Increment counters
    row_count[row_idx] += 1
    col_count[col_idx] += 1
    if row_idx == col_idx:
        main_diag_count += 1
    if row_idx + col_idx == N - 1:
        anti_diag_count += 1
    
    # Check for bingo
    if (row_count[row_idx] == N or
        col_count[col_idx] == N or
        (row_idx == col_idx and main_diag_count == N) or
        (row_idx + col_idx == N - 1 and anti_diag_count == N)):
        bingo_turn = turn
        break

if bingo_turn != -1:
    print(bingo_turn)
else:
    print(-1)