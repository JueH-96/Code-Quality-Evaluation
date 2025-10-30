import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = []
for i in range(N):
    S.append(data[index])
    index += 1
# Reverse S to get row_strings where row_strings[0] is for row 1, etc.
row_strings = S[::-1]
len_row = [len(s) for s in row_strings]
M = max(len_row)
for j in range(1, M + 1):  # j is 1-based column index
    # Find the maximum row index (0-based) where j <= len_row[idx]
    max_row_idx = max(idx for idx in range(N) if j <= len_row[idx])
    # Build the string for T_j
    t_j_str_parts = []
    for p in range(max_row_idx + 1):  # p is row index from 0 to max_row_idx
        if j <= len_row[p]:
            char = row_strings[p][j - 1]  # j-1 is 0-based column index in string
        else:
            char = '*'
        t_j_str_parts.append(char)
    # Join and print the string
    print(''.join(t_j_str_parts))