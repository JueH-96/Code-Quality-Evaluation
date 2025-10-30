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
        P.append(input[ptr])
        ptr += 1

    # Precompute total_row
    total_row = [0] * N
    for r in range(N):
        cnt = 0
        for c in range(N):
            if P[r][c] == 'B':
                cnt += 1
        total_row[r] = cnt

    # Precompute row_total_prefix
    row_total_prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        row_total_prefix[i] = row_total_prefix[i - 1] + total_row[i - 1]
    total_all_B = row_total_prefix[N]

    # Precompute row_sum_col
    row_sum_col = [[0] * (N + 1) for _ in range(N)]
    for r in range(N):
        current = 0
        row_sum_col[r][0] = 0
        for k in range(1, N + 1):
            if P[r][k - 1] == 'B':
                current += 1
            row_sum_col[r][k] = current

    # Precompute col_partial_prefix
    col_partial_prefix = [[0] * (N + 1) for _ in range(N)]
    for b in range(N):
        for k in range(1, N + 1):
            col_partial_prefix[b][k] = col_partial_prefix[b][k - 1] + row_sum_col[k - 1][b]

    def compute_f(x, y):
        if x < 0 or y < 0:
            return 0
        X = x + 1
        Y = y + 1
        full_row = X // N
        rem_row = X % N
        full_col = Y // N
        b = Y % N

        # Compute sum_total_all
        sum_total_all = full_col * total_all_B
        sum_total_all += col_partial_prefix[b][N]

        sum_full_cycles = full_row * sum_total_all

        # Compute sum_partial_rows
        sum_partial_full = full_col * row_total_prefix[rem_row]
        sum_partial_remainder = col_partial_prefix[b][rem_row]
        sum_partial = sum_partial_full + sum_partial_remainder

        total = sum_full_cycles + sum_partial
        return total

    output = []
    for _ in range(Q):
        A = int(input[ptr])
        B = int(input[ptr + 1])
        C = int(input[ptr + 2])
        D = int(input[ptr + 3])
        ptr += 4

        fCD = compute_f(C, D)
        fA1D = compute_f(A - 1, D)
        fCB1 = compute_f(C, B - 1)
        fA1B1 = compute_f(A - 1, B - 1)

        res = fCD - fA1D - fCB1 + fA1B1
        output.append(str(res))

    print('
'.join(output))

if __name__ == "__main__":
    main()