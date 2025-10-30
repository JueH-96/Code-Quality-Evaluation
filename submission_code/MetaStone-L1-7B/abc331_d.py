import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    P = []
    for _ in range(N):
        row = input[ptr]
        ptr += 1
        P.append([1 if c == 'B' else 0 for c in row])

    # Precompute prefix_row for each row
    prefix_row = []
    for r in range(N):
        pr = [0] * (N + 1)
        for c in range(N):
            pr[c+1] = pr[c] + P[r][c]
        prefix_row.append(pr)
    
    B_total = sum(row[-1] for row in prefix_row)

    for _ in range(Q):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        D = int(input[ptr])
        ptr += 1

        R_blocks = (C - A + 1) // N
        R_remain = (C - A + 1) % N

        C_blocks = (D - B + 1) // N
        C_remain = (D - B + 1) % N

        full_blocks = R_blocks * C_blocks * B_total

        sum_partial_rows = 0
        for i in range(R_remain):
            r = (A % N + i) % N
            a = B
            b = D
            if a > b:
                continue
            mod_a = a % N
            mod_b = b % N
            if mod_a <= mod_b:
                sum_r = prefix_row[r][mod_b + 1] - prefix_row[r][mod_a]
            else:
                sum_r = (prefix_row[r][N] - prefix_row[r][mod_a]) + (prefix_row[r][mod_b + 1])
            sum_partial_rows += sum_r

        sum_partial_columns = 0
        for j in range(C_remain):
            c = (B % N + j) % N
            a = A
            b = C
            if a > b:
                continue
            mod_a = a % N
            mod_b = b % N
            if mod_a <= mod_b:
                sum_c = prefix_row[c][mod_b + 1] - prefix_row[c][mod_a]
            else:
                sum_c = (prefix_row[c][N] - prefix_row[c][mod_a]) + (prefix_row[c][mod_b + 1])
            sum_partial_columns += sum_c

        start_row = A + R_blocks * N
        end_row = C
        start_col = B + C_blocks * N
        end_col = D

        if start_row > end_row or start_col > end_col:
            sum_overlap = 0
        else:
            sum_overlap = prefix_row[start_row % N][end_col % N + 1] - prefix_row[start_row % N][start_col % N]
            sum_overlap -= prefix_row[end_row % N][end_col % N + 1] - prefix_row[end_row % N][start_col % N]

        total = full_blocks + sum_partial_rows + sum_partial_columns - sum_overlap
        print(total)

if __name__ == '__main__':
    main()